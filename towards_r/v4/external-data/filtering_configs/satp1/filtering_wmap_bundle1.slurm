#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=20
#SBATCH --ntasks=1120
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --job-name=wmap-satp1-ext-filtering-bundle1
#SBATCH --mail-user arianna.rizzieri@physics.ox.ac.uk

set -e

# Log file
tel_ext_data="wmap"  # planck # wmap
tel_filter="satp1"
bundle="1"   # 0 # 1
log="./logs/log_${tel_ext_data}_${bundle}_${tel_filter}_ext_filtering"

export OMP_NUM_THREADS=2

bb_awg_scripts_dir=/home/ar3186/awg-clean/bb-awg-scripts/ ## YOUR bb_awg_scripts DIRECTORY

filtering_config=${bb_awg_scripts_dir}/towards_r/v4/external-data/filtering_configs/${tel_filter}/filtering_config_${tel_ext_data}_bundle${bundle}.yaml

echo "Launching pipeline at $(date)"

srun -n 1120 -c 2 --cpu_bind=cores \
     python -u ${bb_awg_scripts_dir}/pipeline/filtering/filter_ext_sotodlib.py --config_file $filtering_config \
     > ${log} 2>&1

echo "Ending batch script at $(date)"
