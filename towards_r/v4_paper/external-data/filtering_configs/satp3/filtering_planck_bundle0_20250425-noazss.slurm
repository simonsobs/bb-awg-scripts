#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=20
#SBATCH --ntasks=1120
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --job-name=satp3-filtering-pk-b1
#SBATCH --mail-user sazzoni@princeton.edu

set -e

module use --append /scratch/gpfs/SIMONSOBS/modules                                                                                                                         
module load soconda/3.12/v0.6.8

export PYTHONPATH="${PYTHONPATH}:/home/sa5705/software/sotodlib/"

# Log file
tel_ext_data="planck"  # planck # wmap
tel_filter="satp3"
bundle="1" #"0" 
log="./logs/log_${tel_ext_data}_${bundle}_${tel_filter}_ext_filtering_20260425-noazss.log"

export OMP_NUM_THREADS=2

bb_awg_scripts_dir=/home/sa5705/software/bb-awg-scripts ## YOUR bb_awg_scripts DIRECTORY

filtering_config=${bb_awg_scripts_dir}/towards_r/v4_paper/external-data/filtering_configs/${tel_filter}/filtering_config_${tel_ext_data}_bundle${bundle}_20260425-noazss.yaml

echo "Launching pipeline at $(date)"

srun -n 1120 -c 2 --cpu_bind=cores \
     python -u ${bb_awg_scripts_dir}/pipeline/filtering/filter_ext_sotodlib.py --config_file $filtering_config \
    > ${log} 2>&1

echo "Ending batch script at $(date)"
