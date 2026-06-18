#!/bin/bash

pix_type=hp
nside=64  # Ignored for CAR
res_arcmin=5  # Ignored for HEALPIX
smooth_fwhm=60
n_sims=200
id_sims_start=0
# out_dir=/cephfs/soukdata/user_data/kwolz/simpure/plaw_sims  # SO:UK
out_dir=/pscratch/sd/k/kwolz/bbdev/simpure/plaw_sims  # NERSC
bb_awg_scripts_dir=/global/homes/k/kwolz/bbdev/bb-awg-scripts  # YOUR BB-AWG-SCRIPTS DIR
car_template=${bb_awg_scripts_dir}/pipeline/simpure/band_car_fejer1_20arcmin.fits  # Ignored for HEALPIX

mkdir -p $out_dir

## Generate a set of pure-type simulations
python ${bb_awg_scripts_dir}/pipeline/misc/get_plawTEB_simulations.py \
    --pix_type $pix_type \
    --nside $nside \
    --smooth_fwhm $smooth_fwhm \
    --n_sims $n_sims \
    --sim_id_start $id_sims_start \
    --out_dir $out_dir \
    --car_template_map $car_template \
    --res_arcmin $res_arcmin \
    --pols_keep B