#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=2
#SBATCH --ntasks=22
#SBATCH --cpus-per-task=10
#SBATCH --time=8:00:00
#SBATCH --job-name=wmap-0-satp3-ext-coadd

set -e

# Log file
tel_ext_data="wmap"  # planck # wmap
tel_filter="satp3"
bundle="0"   # 0 # 1
log="./logs/log_${tel_ext_data}_${bundle}_${tel_filter}_ext_coadd"

export OMP_NUM_THREADS=10

bb_awg_scripts_dir=/home/ar3186/awg-clean/bb-awg-scripts ## YOUR bb_awg_scripts DIRECTORY

filtering_config=${bb_awg_scripts_dir}/towards_r/v4/external-data/filtering_configs/${tel_filter}/filtering_config_${tel_ext_data}_bundle${bundle}.yaml

echo "Launching pipeline at $(date)"

srun -n 22 -c 10 --cpu_bind=cores \
     python -u ${bb_awg_scripts_dir}/pipeline/filtering/coadd_filtered_ext.py --config_file $filtering_config \
     > ${log} 2>&1

echo "Ending batch script at $(date)"
