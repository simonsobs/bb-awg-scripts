#!/bin/bash

pix_type="hp"
nside=64  # Ignored for CAR
res_arcmin=20  # Ignored for HEALPIX
nsims=200
n_bundles=2
id_start=0
noise_muK_amin=2
# out_dir=/cephfs/soukdata/user_data/kwolz/simpure/cmb_sims  # SO:UK
out_dir=/pscratch/sd/k/kwolz/bbdev/simpure/noise_sims  # NERSC
bb_awg_scripts_dir=/global/homes/k/kwolz/bbdev/bb-awg-scripts  # YOUR BB-AWG-SCRIPTS DIR
car_template=/global/homes/k/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/data/fullsky_car_fejer1_20arcmin.fits


mkdir -p $out_dir

## Generate a set of CMB simulations
python ${bb_awg_scripts_dir}/pipeline/misc/get_noise_simulations.py \
    --pix_type $pix_type \
    --noise_muK_amin $noise_muK_amin \
    --id_start $id_start \
    --nside $nside \
    --n_bundles $n_bundles \
    --n_sims $nsims \
    --out_dir $out_dir \
    --car_template $car_template