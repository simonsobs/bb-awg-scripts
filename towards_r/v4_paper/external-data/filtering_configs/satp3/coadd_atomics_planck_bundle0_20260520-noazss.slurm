#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=28
#SBATCH --time=8:00:00
#SBATCH --job-name=1-satp3-coadd-pk
#SBATCH --mail-user sazzoni@princeton.edu

set -e

module use --append /scratch/gpfs/SIMONSOBS/modules                                                                                                                         
module load soconda/3.12/v0.6.8

export PYTHONPATH="${PYTHONPATH}:/home/sa5705/software/sotodlib/"

# Log file
tel_ext_data="planck"  # planck # wmap
tel_filter="satp3"
bundle="1"   # 0 # 1
log="./logs/log_${tel_ext_data}_${bundle}_${tel_filter}_ext_coadd_20260520-noazss.log"

export OMP_NUM_THREADS=28

bb_awg_scripts_dir=/home/sa5705/software/bb-awg-scripts ## YOUR bb_awg_scripts DIRECTORY

filtering_config=${bb_awg_scripts_dir}/towards_r/v4_paper/external-data/filtering_configs/${tel_filter}/filtering_config_${tel_ext_data}_bundle${bundle}_20260520.yaml

echo "Launching pipeline at $(date)"

srun -n 16 -c 28 --cpu_bind=cores \
     python -u ${bb_awg_scripts_dir}/pipeline/filtering/coadd_filtered_ext.py --config_file $filtering_config \
     > ${log} 2>&1

echo "Ending batch script at $(date)"
