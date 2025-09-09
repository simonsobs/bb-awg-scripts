#!/bin/bash -l

#SBATCH --nodes=4
#SBATCH --cpus-per-task=112
#SBATCH --time=03:00:00
#SBATCH --job-name=simpure_filtering
#SBATCH --exclusive

set -e

#. setup_cmb_sims.sh  # for CMB sims
 . setup_tf_sims.sh

# Log file
log="./log_simpure_filtering"

#=========== Compute runtime parameters ============
export OMP_NUM_THREADS=2
module use /cephfs/soukdata/software/modulefiles/
module load soconda/20241017_3.10

#=========== Run it ============
echo "Launching pipeline at $(date)"

launch_str="srun -n 117 -c 3 --cpu_bind=cores"
com="${launch_str} python -u ${bb_awg_scripts_dir}/pipeline/filtering/filter_simple.py --config_file $filtering_config"
echo ${com}
eval ${com} > ${log} 2>&1

launch_str="srun -n 30 -c 14 --cpu_bind=cores"
com="${launch_str} python -u ${bb_awg_scripts_dir}/pipeline/filtering/coadd_simple.py --config_file $filtering_config"
echo ${com}
eval ${com} > ${log} 2>&1
echo "Ending batch script at $(date)"