#!/bin/bash

set -e

# Environment
bb_awg_scripts_dir=/home/kw6905/bbdev/bb-awg-scripts  # YOUR bb-awg-scripts DIR

pix_type=car
smooth_fwhm=30
n_sims=20
id_start=0
hi_res=1  # resolution in arcmin for high-res maps
# This templated was introduced with ISO v4
car_template=/scratch/gpfs/SIMONSOBS/so/science-readiness/geometry/sat_f090.fits

## Generate a set of pure-T/E/B simulations for transfer function estimation

## High-resolution maps to learn pixel window function (full sky)
out_dir=/scratch/gpfs/SIMONSOBS/sat-iso/v1/transfer_function/input_sims
mkdir -p $out_dir

## NOTE: the following script has already been run.
## Before rerunning, contact the owners of out_dir.
# python ${bb_awg_scripts_dir}/pipeline/misc/get_tf_simulations.py \
#     --pix_type=${pix_type} \
#     --smooth_fwhm=${smooth_fwhm} \
#     --res_arcmin=${hi_res} \
#     --n_sims=${n_sims} \
#     --sim_id_start=${id_start} \
#     --out_dir=${out_dir}

## Final resolution maps
out_dir=/scratch/gpfs/SIMONSOBS/sat-iso/v4/transfer_function/input_sims
mkdir -p $out_dir

## NOTE: the following script has already been run.
## Before rerunning, contact the owners of out_dir.
# python -u ${bb_awg_scripts_dir}/pipeline/misc/get_tf_simulations.py \
#     --pix_type=$pix_type \
#     --smooth_fwhm=$smooth_fwhm \
#     --n_sims=$n_sims \
#     --sim_id_start=${id_start} \
#     --out_dir=$out_dir \
#     --car_template_map $car_template