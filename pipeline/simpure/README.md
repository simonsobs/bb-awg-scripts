# Simulation-based purification pipeline.

**Main run script: ** `simpure/scripts/iterative_purification.py`

**Purpose:** Compute TF, mode coupling matrix, E-to-B leakage deprojection matrix and 
validate them on CMB-only simulations.

The main script depends on input from:
* `scripts/get_optimal_patch_atomics.sh` - make new atomic DB and new hits map ( only)
* `scripts/obsmat_filter_sims.py` - filter simulations using an observation matrix (HP only)
* `../misc/gen_analysis_mask_simple.py` - make analysis mask (optional)
* `scripts/gen_tf_sims.sh` - make Gaussian power-law simulations for TF estimation and leakage deprojection
* `scripts/gen_cmb_sims.sh` - make validation simulations
