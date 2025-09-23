[![JACS Au](https://img.shields.io/badge/JACS%20Au-d9b644.svg)](https://doi.org/10.1021/jacsau.5c00602)

# Generative Deep Learning Pipeline Yields Potent Gram-Negative Antibiotics

Martin F. Köllen,<sup>#</sup> Maximilian G. Schuh,<sup>#</sup> Robin Kretschmer, Joshua Hesse, Dominik Schum, Junhong Chen, Annkathrin I. Bohne, Dominik P. Halter, Stephan A. Sieber*

(<sup>#</sup> Contributed equally to this work; * Corresponding author)

**Abstract**

The escalating crisis
of multiresistant bacteria demands the rapid discovery of novel antibiotics
that transcend the limitations imposed by the biased chemical space
of current libraries. To address this challenge, we introduce an innovative
deep learning-driven pipeline for de novo antibiotic
design. Our unique approach leverages a chemical language model to
generate structurally unprecedented antibiotic candidates. The model
was trained on a diverse chemical space of drug-like molecules and
natural products. We then applied transfer learning using a data set
of diverse antibiotic scaffolds to refine its generative capabilities.
Using predictive modeling and expert curation, we prioritized the
most promising compounds for synthesis. This pipeline identified a
lead candidate with potent activity against methicillin-resistant *Staphylococcus aureus*. We then performed iterative
refinement by synthesizing 40 derivatives of the lead compound. This
effort produced a suite of active compounds, with 30 showing activity
against *S. aureus* and 17 against *Escherichia coli*. Among these, lead compound **D8** exhibited remarkable submicromolar and single-digit micromolar
potency against the aforementioned pathogens, respectively. Mechanistic
investigations point to the reductive generation of reactive species
as its primary mode of action. This work validates a deep-learning
pipeline that explores chemical space to generate antibiotic candidates.
This process yields a potent nitrofuran derivative and a set of experimentally
validated scaffolds to seed future antibiotic development.

## Structure

```
AIbiotics
│
├── LICENSE                           # License file for project use
├── README.md                         # Project overview and usage instructions
├── env.yml                           # Conda environment
│
├── data/                             # Core data folder containing all inputs and processed datasets
│   ├── entry_curation.ipynb          # Jupyter notebook that curates and filters the eNTRy dataset
│   ├── PubChem_compound_text_antibiotic.csv   # 2239 known antibiotics annotated from PubChem 
│   ├── transfer_learning_antibiotics.txt      # SMILES strings used for chemical language model fine-tuning
│   │
│   ├── derivatives/                  # Molecule sets for automated SAR testing of lead compound derivatives
│   │   ├── 3-(5-Nitro-2-furyl)acrylic acid.txt        # Core antibiotic structure to be derivatized
│   │   └── Combination products without salts.txt     # All possible antibiotic-like reaction products
│   │
│   ├── de_novo_antibiotics/          # Generated molecules using fine-tuned model
│   │   ├── de_novo_pubchem_antibiotic.csv     # All generated SMILES from transfer-learned model
│   │   ├── molecules_10_0.7.txt                 # Epoch 10 generation 
│   │   ├── molecules_20_0.7.txt                 # Epoch 20 generation 
│   │   ├── molecules_30_0.7.txt                 # Epoch 30 generation 
│   │   └── molecules_40_0.7.txt                 # Epoch 40 generation
│   │
│   └── entry_dataset/                # Filtered compound sets for permeability/accumulation (Richter et al.)
│       ├── accumulators_smiles.csv         # Molecules known to accumulate in bacteria
│       ├── non_accumulators_smiles.csv     # Molecules known NOT to accumulate
│       ├── merged_cleaned_dataset.csv      # Combined dataset used for model training
│       ├── table1.csv – table6.csv         # Original datatables (Richter et al.)
│
├── ms_data/                          # MS data processing
│   ├── correlation_analysis.ipynb          # Correlation analysis of the generated MS data
│   ├── diannpar.txt                        # DIA-NN parameter file
│   ├── JH15.txt - JHamp.txt                # Perseus output files
│   └── UP000000625_83333.fasta             # FASTA file of E. coli
│
├── ranking/                          # Scoring pipeline to rank de novo generated molecules
│   ├── de_novo_pipeline.py                 # Main script ranks molecules based on accumulation and synthetic accessibility
│   └── final_antibiotics_predictions.csv   # Output file with top-ranked candidate antibiotics
│
└── sar/                              # Structure–Activity Relationship pipeline to rank derivatives
    ├── sar_pipeline.py               # SAR evaluation script for lead compound derivatives
    └── derivatives_predictions.csv   # Output file with model predictions on SAR derivative compounds

```

## Usage

### Installation

Follow these commands to download and install all scripts and packages.

```bash
git clone https://github.com/sieber-lab/AIbiotics.git
cd AIbiotics

conda env create -f env.yml
conda activate AIbiotics
```

Follow the instructions from this [TwinBooster](https://github.com/maxischuh/TwinBooster) repository to set up TwinBooster correctly and download its weights from the provided links or use `twinbooster.download_models()`.

### *De novo* generation

Please follow the instructions from the [virtual_libraries](https://github.com/ETHmodlab/virtual_libraries) GitHub to generate new molecules.
We used the `data/transfer_learning_antibiotics.txt` file to generate our *de novo* antibiotics. 

### *De novo*-generated molecule scoring pipeline

This is the command line interface to score our *de novo* molecules. 
You can adapt all parameters and input files as you wish.

```
python ranking/de_novo_pipeline.py \
  --input my_smiles.csv \
  --smiles-column SMILES \
  --entry-dataset entry_dataset/merged_cleaned_dataset.csv \
  --output results.csv \
  --ecfp-bits 1024 \
  --top-k 5 \
  --test-size 0.8 \
  --random-state 123 \
  --tb-model-path /models/tb_model \
  --tb-lgbm-path /models/tb_lgbm.joblib \
  --description "Custom description for TwinBooster"
```

### Structure–activity relationship pipeline to rank derivatives

This is the command line interface to rank our derivatives for accumulation. 
You can adapt all parameters and input files as you wish.
```
python sar/sar_pipeline.py \
  -d my_derivatives.csv \               
  --derivatives-smiles-column SMILES \ 
  -e entry_data.csv \
  --entry-label-column Label \
  -o results.csv \
  --label-map low:0,high:1 \
  --fp-types Mordred ECFP \
  --no-mordred-3d \
  --ecfp-bits 2048 \
  --n-jobs 16 \
  --ag-verbosity 2
  ```

## Citation

If you use our work in your research, please cite :)
```tex
@article{kollen2025generative,
  title = {Generative {{Deep Learning Pipeline Yields Potent Gram-Negative Antibiotics}}},
  author = {K{\"o}llen, Martin F. and Schuh, Maximilian G. and Kretschmer, Robin and Hesse, Joshua and Schum, Dominik and Chen, Junhong and Bohne, Annkathrin I. and Halter, Dominik P. and Sieber, Stephan A.},
  year = {2025},
  month = sep,
  journal = {JACS Au},
  volume = {5},
  number = {9},
  pages = {4249--4259},
  publisher = {American Chemical Society},
  doi = {10.1021/jacsau.5c00602},
  urldate = {2025-09-22}
}
```