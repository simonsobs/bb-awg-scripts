#!/bin/bash
## A simple script to just make maps.
## If you have relative paths in the config file you will need to be in that directory

## Load the software stack
module use /global/common/software/sobs/perlmutter/modulefiles
module load soconda

## Change these
export OMP_NUM_THREADS=4
nproc=32 
pname=${PWD}/make_atomic_filterbin_map.py
config=${PWD}/../../users/erosenberg/configs/241031/config_hp_new_satp1.yaml

## Run
python $pname --config $config --nproc $nproc
