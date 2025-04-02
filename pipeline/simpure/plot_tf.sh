#! /bin/bash

tf_dir=/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/transfer_function/satp1_iso_fp_thin32/f090/transfer_functions

label="fp_thin 32"

python plot_tf.py \
    --tf_file "${tf_dir}/transfer_function_SATp1_f090_south_science_x_SATp1_f090_south_science.npz" \
    --binning_file /scratch/gpfs/SIMONSOBS/sat-iso/transfer_function/soopercool_inputs/binning_car_lmax2160_deltal15_custom.npz \
    --tf_label "${label}" \
    --plot_fname "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/transfer_function/tf_comparison.png"