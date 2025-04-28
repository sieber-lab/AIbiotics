[![ChemrXiv](https://img.shields.io/badge/ChemrXiv-ebe000.svg)](https://doi.org/10.26434/chemrxiv-2025-s418c)

# Generative deep learning pipeline yields potent Gram-negative antibiotics

Martin F. Köllen,<sup>#</sup> Maximilian G. Schuh,<sup>#</sup> Robin Kretschmer, Junhong Chen, Annkathrin I. Bohne, Dominik P. Halter, Stephan A. Sieber*

(<sup>#</sup> Contributed equally to this work; * Corresponding author)

**Abstract**

The escalating crisis of multiresistant bacteria demands the rapid discovery of novel antibiotics that transcend the limitations imposed by the biased chemical space of current libraries.
To address this challenge, we introduce an innovative deep learning-driven pipeline for *de novo* antibiotic design.  
This unique approach leverages a chemical language model, trained on a diverse chemical space encompassing drug-like molecules and natural products, coupled with transfer learning on diverse antibiotic scaffolds to efficiently generate structurally unprecedented antibiotic candidates.
Through the use of predictive modeling and expert curation, we prioritized and synthesized the most promising and readily available candidates.  
Notably, our efforts culminated in a lead candidate demonstrating potent activity against methicillin-resistant *Staphylococcus aureus*.
Iterative refinement through automated synthesis of 40 derivatives yielded a suite of active compounds, including 30 with activity against *S. aureus* and 17 against *Escherichia coli*.  
Among these, lead compound **D8** exhibited remarkable submicromolar and single-digit micromolar potency against the aforementioned pathogens, respectively.
Mechanistic investigations point to the generation of radical species as its primary mode of action.  
This work showcases the power of our innovative deep learning framework to significantly accelerate and expand the horizons of antibiotic drug discovery.

## Structure

- `data` contains all structures/files that were used for transfer learning.

- `ranking` contains all scripts/files/structures that where computationally analyzed. 

- `sar` contains all scripts/files/structures for SAR that where computationally analyzed. 

## Citation

If you use our work in your research, please cite:
```
@misc{kollen2025generative,
  title = {Generative Deep Learning Pipeline Yields Potent {{Gram-negative}} Antibiotics},
  author = {K{\"o}llen, Martin F. and Schuh, Maximilian G. and Kretschmer, Robin and Chen, Junhong and Bohne, Annkathrin I. and Halter, Dominik P. and Sieber, Stephan A.},
  year = {2025},
  month = apr,
  publisher = {ChemRxiv},
  doi = {10.26434/chemrxiv-2025-s418c},
  urldate = {2025-04-28},
  abstract = {The escalating crisis of multiresistant bacteria demands the rapid discovery of novel antibiotics that transcend the limitations imposed by the biased chemical space of current libraries. To address this challenge, we introduce an innovative deep learning- driven pipeline for de novo antibiotic design. This unique approach leverages a chemical language model, trained on a diverse chemical space encompassing drug-like molecules and natural products, coupled with transfer learning on diverse antibiotic scaffolds to efficiently generate structurally unprecedented antibiotic candidates. Through the use of predictive modeling and expert curation, we prioritized and synthesized the most promising and readily available candidates. Notably, our efforts culminated in a lead candidate demonstrating potent activity against methicillin-resistant Staphylococcus aureus. Iterative refinement through automated synthesis of 40 derivatives yielded a suite of active compounds, including 30 with activity against S. aureus and 17 against Escherichia coli. Among these, lead compound D8 exhibited remarkable submicromolar and single-digit micromolar potency against the aforementioned pathogens, respectively. Mechanistic investigations point to the generation of radical species as its primary mode of action. This work showcases the power of our innovative deep learning framework to significantly accelerate and expand the horizons of antibiotic drug discovery.},
  archiveprefix = {ChemRxiv},
  langid = {english},
  keywords = {antibiotics,automated synthesis,de novo drug design,deep learning,drug discovery,Gram-negative,machine learning,MRSA},
}

```