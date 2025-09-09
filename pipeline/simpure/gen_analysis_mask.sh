#!/bin/bash

. setup_tf_sims.sh

out_dir=/cephfs/soukdata/user_data/kwolz/simpure/soopercool_inputs
pix_type=hp  # car
car_template=/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/band_car_fejer1_20arcmin.fits
n_bundles=1
map_set=SATp1_f090_south_science
global_hits=/cephfs/soukdata/user_data/kwolz/simpure/soopercool_inputs/masks/hits_1year_200dets_sat_beams.fits
map_dir=/cephfs/soukdata/user_data/kwolz/simpure/filtered_pure_sims/satp3/f090/butter4_cutoff_1e-2/coadded_sims
map_string_format="pureB_20.0arcmin_fwhm30.0_sim0000_CAR_f090_science_filtered_weights.fits"



python ${bb_awg_scripts_dir}/pipeline/misc/get_analysis_mask_simple.py \
    --out_dir $out_dir \
    --pix_type $pix_type \
    --global_hits $global_hits \
    --car_template $car_template \
    --nside 256 \
    --n_bundles $n_bundles \
    --map_set $map_set \
    --map_dir $map_dir \
    --map_string_format $map_string_format \
    --smooth_radius 10 \
    --apod_radius 10 \
    --apod_type C1 \
    --verbose \
    #--decra_minmax -48 -32 -10 130 \
    #--flat_mask \
    
    
