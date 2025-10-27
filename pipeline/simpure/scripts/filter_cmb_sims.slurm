#!/bin/bash -l

#SBATCH --nodes=8
#SBATCH --cpus-per-task=112
#SBATCH --time=24:00:00
#SBATCH --job-name=satp3-simpure
#SBATCH --exclusive

set -e

nnodes=8
ncores=112
natomics=876


# Log file
log="./log_satp3"

bb_awg_scripts_dir=/shared_home/kwolz/bbdev/bb-awg-scripts
filtering_config=${bb_awg_scripts_dir}/pipeline/simpure/satp3/butter4_20251007/filtering_config_cmb_sims_butter4.yaml

rundir=/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure
cd $rundir

#=========== Compute runtime parameters ============
export OMP_NUM_THREADS=1
module use /cephfs/soukdata/software/modulefiles/
module load soconda/20241017_3.10

#=========== Run it ============
echo "Launching pipeline at $(date)"

launch_str="srun -n $(($nnodes * $ncores)) -c 1 --cpu_bind=cores"
python_str="python -u ${bb_awg_scripts_dir}/pipeline/filtering/filter_simple.py --config_file $filtering_config"
echo "${launch_str} ${python_str}"
$launch_str \
$python_str \
> "${log}_filter" 2>&1

launch_str="srun -n $(($nnodes * $ncores / 20)) -c 20  --cpu_bind=cores"
python_str="python -u ${bb_awg_scripts_dir}/pipeline/filtering/coadd_simple.py --config_file $filtering_config"
echo "${launch_str} ${python_str}"
$launch_str \
$python_str \
> "${log}_coadd" 2>&1

srun -n 1 -c 1 --cpu_bind=cores \
python -u ${bb_awg_scripts_dir}/pipeline/filtering/delete_atomic_sims_simple.py --config_file $filtering_config \
> "${log}_delete" 2>&1

echo "Ending batch script at $(date)"
