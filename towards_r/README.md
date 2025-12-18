# towards_r

This repository tracks the analysis steps, scripts, and branches used to bring the **SAT-ISO pipeline** from calibrated power spectra to constraints on the tensor-to-scalar ratio **r**.

It is intended to record:
- which scripts were run and how to run them,
- which branches / versions were used,
- and how intermediate products were combined into final likelihood inputs.

The directory structure mirrors `iso-sat` versions.

---

## Repository structure

towards_r/
├── v3/
│ ├── external-data/
│ ├── to-sacc/
│ └── make-sims/
└── README.md


### `v3/`
Analysis corresponding to **iso-sat v3**.

This directory contains all scripts and configuration files needed to go from v3 SAT-ISO spectra to final inputs for the `r` analysis.

---

### `v3/external-data/`
Scripts to process **external datasets** used in the v3 analysis.

This includes preparation, formatting, and any required transformations of non-SO inputs so that they are compatible with the SAT-ISO pipeline and downstream likelihood code.

---

### `v3/to-sacc/`
Scripts used to construct the **final SACC file** for the `r` analysis.

The output is a *single SACC file* containing:
- EB-nulled SO spectra, and  
- all included external datasets,  

combined in a consistent format ready for likelihood evaluation.

---

### `v3/make-sims/`
Scripts to generate simulations used in the analysis.

This includes:
1. **Pure signal simulations** (T / E / B only), and  
2. **Simulation-based covariance** realizations used to estimate covariance matrices and validate the pipeline.

---

## Notes
- Scripts are expected to evolve as analysis choices are refined; this repo serves as a reproducible record of those choices.
