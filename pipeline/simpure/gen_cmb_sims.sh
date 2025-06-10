#!/bin/bash

pix_type="car"
res_arcmin=5
smooth_fwhm=30
n_sims=10
out_dir=/scratch/gpfs/SIMONSOBS/users/kw6905/cmb_sims  # /scratch/gpfs/SIMONSOBS/sat-iso/
car_template=/home/kw6905/bbdev/pwg-scripts/iso-sat-review/mapmaking/band_car_fejer1_5arcmin.fits

mkdir -p $out_dir

bb_awg_scripts_dir=/home/kw6905/bbdev/bb-awg-scripts
pwg_scripts_dir=/home/kw6905/bbdev/pwg-scripts

python ${bb_awg_scripts_dir}/pipeline/misc/get_cmb_simulations.py \
    --pix_type=${pix_type} \
    --smooth_fwhm=${smooth_fwhm} \
    --id_start 10 \
    --n_sims=${n_sims} \
    --out_dir=${out_dir} \
    --car_template $car_template \
    --pols_keep "B"
