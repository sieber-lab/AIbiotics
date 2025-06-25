#!/usr/bin/env python3
"""
CLI tool to compute SAR accumulation predictions for derivative molecules using AutoGluon.
"""
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from autogluon.tabular import TabularPredictor
from skfp.preprocessing import *
from skfp.fingerprints import MordredFingerprint, ECFPFingerprint, PhysiochemicalPropertiesFingerprint


def smiles2smiles(smiles: str) -> str:
    """Standardize and clean a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict accumulation for derivative molecules via SAR-ML models."
    )
    # I/O options
    parser.add_argument(
        "-d", "--derivatives-file", type=str,
        default="data/derivatives/Combination products without salts.txt",
        help="Input file (TXT/CSV) with derivative SMILES."
    )
    parser.add_argument(
        "--derivatives-smiles-column", type=str, default="smiles",
        help="Column name for SMILES in derivatives file."
    )
    parser.add_argument(
        "-e", "--entry-dataset", type=str,
        default="data/entry_dataset/merged_cleaned_dataset.csv",
        help="CSV with entry SMILES and accumulation labels."
    )
    parser.add_argument(
        "--entry-smiles-column", type=str, default="smiles",
        help="Column name for SMILES in entry dataset."
    )
    parser.add_argument(
        "--entry-label-column", type=str, default="Accum_class",
        help="Column name for accumulation class in entry dataset."
    )
    parser.add_argument(
        "-o", "--output", type=str,
        default="sar/derivatives_predictions.csv",
        help="Output CSV file path for predictions."
    )
    # Label mapping
    parser.add_argument(
        "--label-map", type=str,
        default="low:0,high:1",
        help="Mapping for accumulation class labels (e.g. low:0,high:1)."
    )
    # Fingerprint settings
    parser.add_argument(
        "--fp-types", nargs="+", choices=["Mordred", "ECFP", "Physiochemical"],
        default=["Mordred", "ECFP", "Physiochemical"],
        help="Fingerprint types to generate."
    )
    parser.add_argument(
        "--ecfp-bits", type=int, default=1024,
        help="Number of bits for ECFP and Physicochemical fingerprints."
    )
    parser.add_argument(
        "--mordred-3d", dest="mordred_3d", action="store_true", default=True,
        help="Enable 3D descriptors for Mordred fingerprints (default on)."
    )
    parser.add_argument(
        "--no-mordred-3d", dest="mordred_3d", action="store_false",
        help="Disable 3D descriptors for Mordred fingerprints."
    )
    parser.add_argument(
        "--n-jobs", type=int, default=32,
        help="Number of parallel jobs for fingerprint generation."
    )
    # Modeling options
    parser.add_argument(
        "--ag-verbosity", type=int, default=0,
        help="Verbosity level for AutoGluon training."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load entry dataset
    entry = pd.read_csv(args.entry_dataset, dtype=object)
    entry = entry[[args.entry_smiles_column, args.entry_label_column]].copy()
    entry.columns = ["smiles", "label"]

    # Map labels
    label_map = dict(item.split(":") for item in args.label_map.split(","))
    entry["label"] = entry["label"].map(lambda x: int(label_map.get(x, x)))

    # Clean SMILES
    entry["smiles"] = entry["smiles"].apply(smiles2smiles)
    entry.dropna(subset=["smiles", "label"], inplace=True)
    entry.drop_duplicates(subset=["smiles"], inplace=True)
    entry.reset_index(drop=True, inplace=True)

    # Load derivatives
    deriv = pd.read_csv(args.derivatives_file, dtype=object)
    deriv = deriv[[args.derivatives_smiles_column]].copy()
    deriv.columns = ["smiles"]
    deriv["smiles"] = deriv["smiles"].apply(smiles2smiles)
    deriv.dropna(subset=["smiles"], inplace=True)
    deriv.drop_duplicates(subset=["smiles"], inplace=True)
    deriv.reset_index(drop=True, inplace=True)

    # Initialize fingerprint generators
    fp_gen = {}
    if "Mordred" in args.fp_types:
        fp_gen["Mordred"] = MordredFingerprint(use_3D=args.mordred_3d, n_jobs=args.n_jobs)
    if "ECFP" in args.fp_types:
        fp_gen["ECFP"] = ECFPFingerprint(fp_size=args.ecfp_bits, n_jobs=args.n_jobs)
    if "Physiochemical" in args.fp-types:
        fp_gen["Physiochemical"] = PhysiochemicalPropertiesFingerprint(fp_size=args.ecfp_bits, n_jobs=args.n_jobs)

    # Generate fingerprints
    data_fp = {}
    deriv_fp = {}
    for name, gen in fp_gen.items():
        data_fp[name] = gen.transform(entry["smiles"])
        deriv_fp[name] = gen.transform(deriv["smiles"])
        assert data_fp[name].shape[0] == entry.shape[0]
        assert deriv_fp[name].shape[0] == deriv.shape[0]

    # Train and predict
    for name in fp_gen.keys():
        print(f"Processing fingerprint: {name}")
        train_df = pd.DataFrame(data_fp[name])
        train_df["label"] = entry["label"]

        predictor = TabularPredictor(
            label="label", eval_metric="average_precision", verbosity=args.ag_verbosity
        ).fit(train_df)

        pred_df = pd.DataFrame(deriv_fp[name])
        probs = predictor.predict_proba(pred_df)
        assert probs.isna().sum().sum() == 0

        deriv[name] = probs[1]

    # Combine
    deriv["total"] = deriv[list(fp_gen.keys())].mean(axis=1)
    deriv.sort_values("total", ascending=False, inplace=True)

    # Save
    deriv.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
