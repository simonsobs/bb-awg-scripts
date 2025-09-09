#!/bin/bash

pix_type="car"
res_arcmin=20
smooth_fwhm=30
n_sims=100
out_dir=/cephfs/soukdata/user_data/kwolz/simpure/input_sims  # /scratch/gpfs/SIMONSOBS/sat-iso/

mkdir -p $out_dir

bb_awg_scripts_dir=/shared_home/kwolz/bbdev/bb-awg-scripts
pwg_scripts_dir=/shared_home/kwolz/bbdev/pwg-scripts

python ${bb_awg_scripts_dir}/pipeline/misc/get_tf_simulations.py \
    --pix_type=${pix_type} \
    --smooth_fwhm=${smooth_fwhm} \
    --n_sims=${n_sims} \
    --out_dir=${out_dir} \
    --car_template_map ${bb_awg_scripts_dir}/pipeline/simpure/band_car_fejer1_20arcmin.fits \
    --res_arcmin $res_arcmin \
    --no_plots
