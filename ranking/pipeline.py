import pandas as pd
import numpy as np

from chembl_structure_pipeline import standardizer

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split

from autogluon.tabular import TabularPredictor

from RAscore import RAscore_NN, RAscore_XGB
import twinbooster

# Initialize RAscore models
nn_scorer = RAscore_NN.RAScorerNN()
xgb_scorer = RAscore_XGB.RAScorerXGB()

# Initialize TwinBooster
tb = twinbooster.TwinBooster(
    model_path="../../twinbooster/scripts/barlow_twins/stash/17112023_2320",
    lgbm_path="../../twinbooster/scripts/lgbm/results/15122023_1758/bt_zero_shot_model_24102023_2058_15122023_1758.joblib"
)

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
    except:
        SMILES_CACHE[smiles] = None
        return None

def weighted_prediction(pred1, pred2, weight):
    """Combine two predictions with a given similarity-based weight."""
    return weight * pred1 + (1 - weight) * pred2


def pairwise_similarity(fp, fp_list, top_k=3):
    """Calculate mean similarity between one fingerprint and the top K fingerprints from a list."""
    similarities = [DataStructs.TanimotoSimilarity(fp, fp2) for fp2 in fp_list]
    return np.mean(sorted(similarities, reverse=True)[:top_k])


def calculate_accumulation_autogluon(df):
    """Calculate accumulation using AutoML (AutoGluon)."""
    data = pd.read_csv("entry_dataset/merged_cleaned_dataset.csv")[['smiles', 'Accum_class']]
    data.rename(columns={'Accum_class': 'label'}, inplace=True)
    data['label'] = data['label'].map({'low': 0, 'high': 1})
    
    data['smiles'] = data['smiles'].apply(get_clean_smiles)
    data.dropna(subset=['smiles'], inplace=True)

    train_smiles, test_smiles, train_labels, test_labels = train_test_split(
        data['smiles'], data['label'], test_size=0.9, random_state=42
    )

    train_fps = np.vstack(train_smiles.apply(get_ecfp4_fingerprints).dropna())
    test_fps = np.vstack(test_smiles.apply(get_ecfp4_fingerprints).dropna())

    antibiotics_fps = np.vstack(df['smiles'].apply(get_ecfp4_fingerprints).dropna())

    train_df = pd.DataFrame(train_fps)
    train_df['label'] = train_labels.values[:len(train_df)]

    test_df = pd.DataFrame(test_fps)
    test_df['label'] = test_labels.values[:len(test_df)]

    predictor = TabularPredictor(label='label', eval_metric='average_precision').fit(train_df, test_df)

    return predictor.predict_proba(pd.DataFrame(antibiotics_fps))[1].values


def calculate_accumulation_twinbooster(smiles_list):
    """Predict accumulation using TwinBooster model."""
    description = "Accumulation of drugs in Gram-negative bacteria using LC–MS/MS as described in provided protocol."
    predictions, confidences = tb.predict(smiles_list, description, get_confidence=True)
    return predictions, confidences


def calculate_pairwise_similarity(df):
    """Compute pairwise similarity between antibiotic and entry dataset molecules."""
    entry = pd.read_csv("entry_dataset/merged_cleaned_dataset.csv")
    entry_fps = [get_ecfp4_fingerprints(smile) for smile in entry['smiles'] if get_ecfp4_fingerprints(smile) is not None]
    antibiotic_fps = [get_ecfp4_fingerprints(smile) for smile in df['smiles'] if get_ecfp4_fingerprints(smile) is not None]

    return np.array([pairwise_similarity(fp, entry_fps, top_k=3) for fp in antibiotic_fps])


def mean_ra_score(smiles):
    """Compute the mean RAscore from NN and XGB models."""
    nn_score = nn_scorer.predict(smiles)
    xgb_score = xgb_scorer.predict(smiles)

    scores = [score for score in [nn_score, xgb_score] if score is not None]

    if not scores:
        return None

    return np.mean(scores)


def main():
    """Main function orchestrating the calculation and combination of predictions."""
    antibiotics = pd.read_csv("data/de_novo_antibiotics/de_novo_pubchem_antibiotic.csv").rename(columns={"Smiles": "smiles"})

    # AutoML predictions
    antibiotics["accumulation_autogluon"] = calculate_accumulation_autogluon(antibiotics)

    # RAscore predictions
    antibiotics["RAscore"] = antibiotics['smiles'].apply(mean_ra_score)

    # TwinBooster predictions
    tb_pred, tb_conf = calculate_accumulation_twinbooster(antibiotics['smiles'].tolist())
    antibiotics["accumulation_twinbooster"] = tb_pred
    antibiotics["confidence_twinbooster"] = tb_conf

    # Similarity weighting
    sim_weights = calculate_pairwise_similarity(antibiotics)
    antibiotics["weighted_accumulation"] = weighted_prediction(
        antibiotics["accumulation_autogluon"],
        antibiotics["accumulation_twinbooster"],
        sim_weights
    )

    # Output the final DataFrame
    antibiotics.to_csv("final_antibiotics_predictions.csv", index=False)


if __name__ == "__main__":
    main()
