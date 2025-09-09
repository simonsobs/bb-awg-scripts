#!/bin/bash

pix_type="hp"
nside=64
res_arcmin=5
smooth_fwhm=30
nsims=100
out_dir=/cephfs/soukdata/user_data/kwolz/simpure/cmb_sims  # /scratch/gpfs/SIMONSOBS/sat-iso/

mkdir -p $out_dir

bb_awg_scripts_dir=/shared_home/kwolz/bbdev/bb-awg-scripts
pwg_scripts_dir=/shared_home/kwolz/bbdev/pwg-scripts

python ${bb_awg_scripts_dir}/pipeline/misc/get_cmb_simulations.py \
    --pix_type=${pix_type} \
    --smooth_fwhm=${smooth_fwhm} \
    --id_start 0 \
    --nside $nside \
    --n_sims $nsims \
    --out_dir $out_dir \
    --car_template ${bb_awg_scripts_dir}/pipeline/simpure/band_car_fejer1_20arcmin.fits \
    --pols_keep "EB"
