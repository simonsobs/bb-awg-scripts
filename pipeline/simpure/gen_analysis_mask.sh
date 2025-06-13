#!/bin/bash

. setup_tf_sims.sh

out_dir=/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/soopercool_inputs
pix_type=car
car_template=/home/kw6905/bbdev/bb-awg-scripts/pipeline/simpure/band_car_fejer1_20arcmin.fits
n_bundles=1
map_set=SATp1_f090_south_science
map_dir=/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/transfer_function/butter4_cutoff_1e-2/coadded_sims
map_string_format="pureB_20.0arcmin_fwhm30.0_sim0000_CAR_f090_science_filtered_weights.fits"



python ${bb_awg_scripts_dir}/pipeline/misc/get_analysis_mask_simple.py \
    --out_dir $out_dir \
    --pix_type $pix_type \
    --car_template $car_template \
    --n_bundles $n_bundles \
    --map_set $map_set \
    --map_dir $map_dir \
    --map_string_format $map_string_format \
    --apod_radius 10 \
    --apod_type C1 \
    --verbose \
    --decra_minmax -47 -33 -10 130 \
