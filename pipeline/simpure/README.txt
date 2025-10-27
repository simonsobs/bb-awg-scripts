Simulation-based purification pipeline.

Main script: iterative_purification.py
Purpose: Compute TF, mode coupling matrix, E-to-B leakage deprojection matrix and 
validate them on CMB-only simulations.

The main script depends on input from:
* scripts/get_optimal_patch_atomics.sh - make new atomic DB and new hits map
* scripts/gen_analysis_mask.sh - make analysis mask
* misc/gen_cl_binning.sh - make multipole binning for power spectra
* scripts/gen_tf_sims.sh - make Gaussian power-law simulations for TF estimation
* scripts/gen_dep_sims.sh - make Gaussian power-law simulations for E-to-B leakage deprojection
* scripts/gen_cmb_sims.sh - make validation simulations
