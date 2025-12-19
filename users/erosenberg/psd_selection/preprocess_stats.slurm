#!/bin/bash

#SBATCH --account=simonsobs
#SBATCH --job-name=preprocess_stats_satp1
#SBATCH --output preprocess_stats_satp1.out
#SBATCH --time 24:00:00
#SBATCH --ntasks=112
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

module use --append /scratch/gpfs/SIMONSOBS/modules
module load soconda/3.12/v0.6.8

export OMP_NUM_THREADS=1

preprocess_archive="/scratch/gpfs/SIMONSOBS/sat-iso/v3/preprocessing/satp1_20250801_init/process_archive.sqlite"
preprocess_config="preprocessing/satp1/preprocessing_config_20250801_init.yaml"
savename="/scratch/gpfs/SIMONSOBS/users/erosenberg/psd_info/psd_info_v3_satp1.npy"
scriptdir=$PWD  # Directory containing get_psd_info.py

# The context file has to be in the place specified by the preprocess config.
cd /home/er5768/iso-sat/v3/

srun -n 112 -c 1 python -u ${scriptdir}/get_psd_info.py \
        ${preprocess_archive} \
        ${preprocess_config} \
        --errlog-ext get_psd_info_err_satp1.txt \
        --savename ${savename} \
        --nproc 112
        
