#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
##SBATCH -q regular
#SBATCH -J make-atomics
#SBATCH --mail-user=erik.rosenberg@manchester.ac.uk
#SBTACH --mail-type=ALL
#SBATCH -A mp107b
##SBATCH -t 04:0:0
#SBATCH -t 0:30:0

export OMP_NUM_THREADS=4
nproc=32
script_dir=/global/homes/r/rosenber/so_pwg/bb-awg-scripts/pipeline/mapmaking/
echo $script_dir
pname=$script_dir
config=$script_dir/../../users/erosenberg/configs/241031/config_hp_satp1_debug.yaml

bash $script_dir/run_mapmaker.sh $pname $config $nproc
