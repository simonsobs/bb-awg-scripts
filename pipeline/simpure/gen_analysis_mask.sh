#!/bin/bash

. setup_tf_sims.sh

out_dir=/cephfs/soukdata/user_data/kwolz/simpure/soopercool_inputs  ## YOUR OUTPUT DIR
pix_type=hp  # car
car_template=/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/band_car_fejer1_20arcmin.fits
n_bundles=1
map_set=SATp1_f090_south_science  ## label (for soopercool) 
global_hits=/cephfs/soukdata/user_data/kwolz/simpure/soopercool_inputs/masks/hits_1year_200dets_sat_beams.fits  ## hits map to build mask from
map_dir=/cephfs/soukdata/user_data/kwolz/simpure/filtered_pure_sims/satp3/f090/butter4_cutoff_1e-2/coadded_sims  # dir of input bundled maps
map_string_format="pureB_20.0arcmin_fwhm30.0_sim0000_CAR_f090_science_filtered_weights.fits"  # name of input bundled maps


## Make apodized analysis mask from hits
## Like soopercool's get_analysis mask.py, but without its dependence on the soopercool config
## Note: map_dir and map_string_format are ignored if global_hits is provided.
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
    --verbose
