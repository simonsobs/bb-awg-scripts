#!/bin/bash

pix_type="hp"
pols_keep="B"  # "EB" means the CMB simulation contains both E-and B-modes
nside=64  # Ignored for CAR
res_arcmin=5  # Ignored for HEALPIX
smooth_fwhm=60
nsims=200
id_start=0
# out_dir=/cephfs/soukdata/user_data/kwolz/simpure/cmb_sims  # SO:UK
out_dir=/pscratch/sd/k/kwolz/bbdev/simpure/cmb_sims  # NERSC
bb_awg_scripts_dir=/global/homes/k/kwolz/bbdev/bb-awg-scripts  # YOUR BB-AWG-SCRIPTS DIR
car_template=${bb_awg_scripts_dir}/pipeline/simpure/band_car_fejer1_20arcmin.fits  # Ignored for HEALPIX

mkdir -p $out_dir

## Generate a set of CMB simulations
python ${bb_awg_scripts_dir}/pipeline/misc/get_cmb_simulations.py \
    --pix_type $pix_type \
    --smooth_fwhm $smooth_fwhm \
    --id_start $id_start \
    --nside $nside \
    --n_sims $nsims \
    --out_dir $out_dir \
    --car_template $car_template \
    --pols_keep $pols_keep
