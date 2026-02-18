#!/bin/bash -l

#SBATCH --nodes=4
#SBATCH --ntasks=448
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --job-name=cmb-filter
#SBATCH --exclusive
#SBATCH --mail-user=kevin.wolz@physics.ox.ac.uk
#SBATCH --mail-type=ALL

set -e

nnodes=4
ncores=112
natomics=876


# Log file
log="/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/scripts/log_cmb_filter"

bb_awg_scripts_dir=/shared_home/kwolz/bbdev/bb-awg-scripts
filtering_config=${bb_awg_scripts_dir}/pipeline/simpure/satp3/poly1_20260203/filtering_config_cmb_sims.yaml

rundir=/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure
cd $rundir

#=========== Compute runtime parameters ============
export OMP_NUM_THREADS=1
module use /cephfs/soukdata/software/modulefiles/
module load soconda/20241017_3.10

#=========== Run it ============
echo "Launching pipeline at $(date)"

launch_str="srun -n $(($nnodes * $ncores)) -c 2 --cpu_bind=cores"
python_str="python -u ${bb_awg_scripts_dir}/pipeline/filtering/filter_simple.py --config_file $filtering_config"
echo "${launch_str} ${python_str}"
$launch_str \
$python_str \
> "${log}" 2>&1

echo "Ending batch script at $(date)"
