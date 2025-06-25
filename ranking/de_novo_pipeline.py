#!/usr/bin/env python3
"""
CLI tool to predict antibiotic accumulation and scores using AutoGluon, RAscore, and TwinBooster.
"""
import argparse
import pandas as pd
import numpy as np

from chembl_structure_pipeline import standardizer
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from RAscore import RAscore_NN, RAscore_XGB
import twinbooster

SMILES_CACHE = {}
SMILES_STR_COL = "smiles"

def get_ecfp4_fingerprints(smiles, n_bits=1024):
    """Convert SMILES strings into ECFP4 fingerprints."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))


def get_clean_smiles(smiles):
    if smiles in SMILES_CACHE:
        return SMILES_CACHE[smiles]
    try:
        mol = Chem.MolFromSmiles(smiles)
        molblock = Chem.MolToMolBlock(mol)
        std_molblock = standardizer.standardize_molblock(molblock)
        parent_molblock, _ = standardizer.get_parent_molblock(std_molblock)
        parent_mol = Chem.MolFromMolBlock(parent_molblock)
        clean_smiles = Chem.MolToSmiles(parent_mol)
        SMILES_CACHE[smiles] = clean_smiles
        return clean_smiles
    except Exception:
        SMILES_CACHE[smiles] = None
        return None


def weighted_prediction(pred1, pred2, weight):
    """Combine two predictions with a given similarity-based weight."""
    return weight * pred1 + (1 - weight) * pred2


def pairwise_similarity(fp, fp_list, top_k=3):
    """Calculate mean similarity between one fingerprint and the top K fingerprints from a list."""
    similarities = [DataStructs.TanimotoSimilarity(fp, fp2) for fp2 in fp_list]
    return np.mean(sorted(similarities, reverse=True)[:top_k])


def calculate_accumulation_autogluon(df, entry_dataset_path, test_size, random_state, ecfp_bits):
    """Calculate accumulation using AutoML (AutoGluon)."""
    data = pd.read_csv(entry_dataset_path)[["smiles", "Accum_class"]]
    data.rename(columns={"Accum_class": "label"}, inplace=True)
    data["label"] = data["label"].map({"low": 0, "high": 1})

    data["smiles"] = data["smiles"].apply(get_clean_smiles)
    data.dropna(subset=["smiles"], inplace=True)

    train_smiles, test_smiles, train_labels, test_labels = train_test_split(
        data["smiles"], data["label"], test_size=test_size, random_state=random_state
    )

    train_fps = np.vstack(train_smiles.apply(lambda x: get_ecfp4_fingerprints(x, n_bits=ecfp_bits)).dropna())
    test_fps = np.vstack(test_smiles.apply(lambda x: get_ecfp4_fingerprints(x, n_bits=ecfp_bits)).dropna())

    antibiotics_fps = np.vstack(df["smiles"].apply(lambda x: get_ecfp4_fingerprints(x, n_bits=ecfp_bits)).dropna())

    train_df = pd.DataFrame(train_fps)
    train_df["label"] = train_labels.values[:len(train_df)]

    test_df = pd.DataFrame(test_fps)
    test_df["label"] = test_labels.values[:len(test_df)]

    predictor = TabularPredictor(label="label", eval_metric="average_precision")\
        .fit(train_df, tune_test_data=test_df)

    return predictor.predict_proba(pd.DataFrame(antibiotics_fps))[1].values


def calculate_accumulation_twinbooster(smiles_list, tb_model, description):
    """Predict accumulation using TwinBooster model."""
    predictions, confidences = tb_model.predict(smiles_list, description, get_confidence=True)
    return predictions, confidences


def calculate_pairwise_similarity(df, entry_dataset_path, ecfp_bits, top_k):
    """Compute pairwise similarity between antibiotic and entry dataset molecules."""
    entry = pd.read_csv(entry_dataset_path)
    entry_fps = [get_ecfp4_fingerprints(s, n_bits=ecfp_bits) for s in entry["smiles"]]
    entry_fps = [fp for fp in entry_fps if fp is not None]

    antibiotic_fps = [get_ecfp4_fingerprints(s, n_bits=ecfp_bits) for s in df["smiles"]]
    antibiotic_fps = [fp for fp in antibiotic_fps if fp is not None]

    return np.array([pairwise_similarity(fp, entry_fps, top_k=top_k) for fp in antibiotic_fps])


def mean_ra_score(smiles, nn_scorer, xgb_scorer):
    """Compute the mean RAscore from NN and XGB models."""
    nn_score = nn_scorer.predict(smiles)
    xgb_score = xgb_scorer.predict(smiles)
    scores = [s for s in (nn_score, xgb_score) if s is not None]
    return np.mean(scores) if scores else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict antibiotic accumulation and scores for a SMILES dataset."
    )
    parser.add_argument(
        "-i", "--input", type=str,
        default="data/de_novo_antibiotics/de_novo_pubchem_antibiotic.csv",
        help="Input CSV file containing SMILES."
    )
    parser.add_argument(
        "--smiles-column", type=str, default="Smiles",
        help="Column name for SMILES in input file."
    )
    parser.add_argument(
        "-e", "--entry-dataset", type=str,
        default="data/entry_dataset/merged_cleaned_dataset.csv",
        help="Entry dataset CSV for AutoGluon and similarity."
    )
    parser.add_argument(
        "-o", "--output", type=str,
        default="ranking/final_antibiotics_predictions.csv",
        help="Output CSV file for predictions."
    )
    parser.add_argument(
        "--ecfp-bits", type=int, default=1024,
        help="Number of bits for ECFP4 fingerprints."
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Top K for similarity calculations."
    )
    parser.add_argument(
        "--test-size", type=float, default=0.9,
        help="Fraction for test split in AutoGluon."
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random state for train-test split."
    )
    parser.add_argument(
        "--tb-model-path", type=str,
        default="../../twinbooster/scripts/barlow_twins/stash/17112023_2320",
        help="Path to TwinBooster model."
    )
    parser.add_argument(
        "--tb-lgbm-path", type=str,
        default="../../twinbooster/scripts/lgbm/results/15122023_1758/bt_zero_shot_model_24102023_2058_15122023_1758.joblib",
        help="Path to TwinBooster LGBM model."
    )
    parser.add_argument(
        "--description", type=str,
        default="Accumulation of drugs in Gram-negative bacteria using LC–MS/MS as described in provided protocol.",
        help="Description for TwinBooster predictions."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load input data
    antibiotics = pd.read_csv(args.input)
    antibiotics.rename(columns={args.smiles_column: "smiles"}, inplace=True)

    # Initialize models
    nn_scorer = RAscore_NN.RAScorerNN()
    xgb_scorer = RAscore_XGB.RAScorerXGB()
    tb_model = twinbooster.TwinBooster(
        model_path=args.tb_model_path,
        lgbm_path=args.tb_lgbm_path
    )

    # AutoGluon predictions
    antibiotics["accumulation_autogluon"] = calculate_accumulation_autogluon(
        antibiotics, args.entry_dataset,
        args.test_size, args.random_state, args.ecfp_bits
    )

    # RAscore predictions
    antibiotics["RAscore"] = antibiotics["smiles"]\
        .apply(lambda s: mean_ra_score(s, nn_scorer, xgb_scorer))

    # TwinBooster predictions
    tb_preds, tb_confs = calculate_accumulation_twinbooster(
        antibiotics["smiles"].tolist(), tb_model, args.description
    )
    antibiotics["accumulation_twinbooster"] = tb_preds
    antibiotics["confidence_twinbooster"] = tb_confs

    # Similarity weights
    sim_weights = calculate_pairwise_similarity(
        antibiotics, args.entry_dataset, args.ecfp_bits, args.top_k
    )
    antibiotics["weighted_accumulation"] = weighted_prediction(
        antibiotics["accumulation_autogluon"],
        antibiotics["accumulation_twinbooster"],
        sim_weights
    )

    # Save results
    antibiotics.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
