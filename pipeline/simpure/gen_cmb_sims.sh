#!/bin/bash

pix_type="car"
res_arcmin=20
smooth_fwhm=30
n_sims=100
out_dir=/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/cmb_sims  # /scratch/gpfs/SIMONSOBS/sat-iso/

mkdir -p $out_dir

bb_awg_scripts_dir=/home/kw6905/bbdev/bb-awg-scripts
pwg_scripts_dir=/home/kw6905/bbdev/pwg-scripts

python ${bb_awg_scripts_dir}/pipeline/misc/get_cmb_simulations.py \
    --pix_type=${pix_type} \
    --smooth_fwhm=${smooth_fwhm} \
    --n_sims=${n_sims} \
    --out_dir=${out_dir} \
    --car_template ${bb_awg_scripts_dir}/pipeline/simpure/band_car_fejer1_20arcmin.fits \
    --pols_keep "E"
