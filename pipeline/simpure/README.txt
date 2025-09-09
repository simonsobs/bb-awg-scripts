Simulation-based purification pipeline.

Main script: iterative_purification.py
Purpose: Computes TF, mode coupling matrix, E-to-B leakage deprojection matrix and 
validates it on CMB-only simulations.

The main script depends on input from:
* get_optimal_patch_atomics.sh - make new atomic DB and new hits map
* gen_analysis_mask.sh - make analysis mask
* gen_mask_mode_coupling.sh - make mask-based mode coupling matrix
* gen_cl_binning.sh - make multipole binning for power spectra
* gen_tf_sims.sh - make power-law simulations for TF estimation and purification
* gen_cmb_sims.sh - make validation simulations
