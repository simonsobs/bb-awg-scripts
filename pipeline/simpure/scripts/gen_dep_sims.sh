#!/bin/bash

pix_type=hp
nside=64
res_arcmin=20
smooth_fwhm=60
n_sims=1000
id_sims_start=0

mkdir -p $out_dir

# out_dir=/cephfs/soukdata/user_data/kwolz/simpure/input_sims  # SO:UK
out_dir=/pscratch/sd/k/kwolz/bbdev/simpure/input_sims  # NERSC
# bb_awg_scripts_dir=/shared_home/kwolz/bbdev/bb-awg-scripts  # SO:UK
bb_awg_scripts_dir=/global/homes/k/kwolz/bbdev/bb-awg-scripts

set OMP_NUM_THREADS=1

## Generate a set of pure-E simulations needed for E-to-B leakage deprojection
# srun -n 112 -c 1 --cpu_bind=cores \
python ${bb_awg_scripts_dir}/pipeline/misc/get_dep_simulations.py \
    --pix_type=$pix_type \
    --nside=$nside \
    --smooth_fwhm=$smooth_fwhm \
    --n_sims=$n_sims \
    --id_sims_start $id_sims_start \
    --out_dir=$out_dir \
    --car_template_map "${bb_awg_scripts_dir}/pipeline/simpure/data/band_car_fejer1_20arcmin.fits" \
    --res_arcmin $res_arcmin