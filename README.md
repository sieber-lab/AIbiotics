# Code and Data for AIbiotics

## An End-to-End Deep Learning Pipeline for Rapid *De Novo* Antibiotic Discovery

Martin F. Köllen,<sup>#</sup> Maximilian G. Schuh,<sup>#</sup> Robin Kretschmer, Junhong Chen, Annkathrin I. Bohne, Dominik P. Halter, Stephan A. Sieber*

(<sup>#</sup> Contributed equally to this work; * Corresponding author)

**Abstract**

To combat the growing threat of antimicrobial resistance, the rapid discovery of novel antibiotics beyond the already existing chemical libraries is required.
We present a deep learning-driven pipeline for *de novo* antibiotic design. 
It uses a chemical language model trained on various molecules---such as drug-like compounds, and natural products---and then applies transfer learning with diverse antibiotic scaffolds to generate structurally unprecedented antibiotic candidates.
The most promising and accessible candidates are selected through the use of predictive models and expert curation, then realized *via* synthesis.
Notably, the most promising candidate synthesized exhibited potent activity against methicillin-resistant *Staphylococcus aureus*.
Automated synthesis of 40 derivatives refined this antibiotic, yielding 30 active compounds against *S. aureus* and 17 active compounds against *Escherichia coli*, including lead compound **D8** with submicromolar and single-digit micromolar potency for the aforementioned bacteria, respectively.
Mechanistic studies suggest the induction of oxidative stress as main mode of action. 
Our approach demonstrates how the power of modern deep learning techniques accelerates and scales antibiotic drug discovery.

---

## Code and Structure

- `data` contains all structures/files that were used for transfer learning.

- `ranking` contains all scripts/files/structures that where computationally analyzed. 

- `sar` contains all scripts/files/structures for SAR that where computationally analyzed. 
