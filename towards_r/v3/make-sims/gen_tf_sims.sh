#!/bin/bash

set -e

# Environment
bb_awg_scripts_dir=/home/kw6905/bbdev/bb-awg-scripts  # YOUR bb-awg-scripts DIR

pix_type=car
smooth_fwhm=30
id_start=0
n_sims=10
hi_res=1  # resolution in arcmin for high-res maps
# This template was used for ISO v2 and v3
car_template=/scratch/gpfs/SIMONSOBS/so/science-readiness/footprint/v20250306/so_geometry_v20250306_sat_f090.fits

## Generate a set of pure-T/E/B simulations for transfer function estimation

## High-resolution maps to learn pixel window function (full sky)
out_dir=/scratch/gpfs/SIMONSOBS/sat-iso/v1/transfer_function/input_sims
mkdir -p $out_dir

## NOTE: the following script has already been run.
## Only rerun these scripts after contacting the owners of out_dir.
# python ${bb_awg_scripts_dir}/pipeline/misc/get_tf_simulations.py \
#     --pix_type=${pix_type} \
#     --smooth_fwhm=${smooth_fwhm} \
#     --res_arcmin=${hi_res} \
#     --n_sims=${n_sims} \
#     --sim_id_start=${id_start} \
#     --out_dir=${out_dir}

## Final resolution maps
out_dir=/scratch/gpfs/SIMONSOBS/sat-iso/v2/tf_sims
mkdir -p $out_dir

## NOTE: the following script has already been run.
## Only rerun these scripts after contacting the owners of out_dir.
# python -u ${bb_awg_scripts_dir}/pipeline/misc/get_tf_simulations.py \
#     --pix_type=$pix_type \
#     --smooth_fwhm=$smooth_fwhm \
#     --n_sims=$n_sims \
#     --sim_id_start=${id_start} \
#     --out_dir=$out_dir \
#     --car_template_map $car_template
