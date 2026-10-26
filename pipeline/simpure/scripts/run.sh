#!/bin/bash -l

#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH -C cpu  # for NERSC
#SBATCH -q regular  # for
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=simpure
#SBATCH --exclusive
#SBATCH --mail-user=kevin.wolz@physics.ox.ac.uk
#SBATCH --mail-type=ALL

set -e


# Log file
log=/global/homes/k/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/scripts/log_simpure  # NERSC
# log=/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/scripts  # SO:UK

rundir=/global/homes/k/kwolz/bbdev/bb-awg-scripts/pipeline/simpure  # NERSC
# rundir=/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/scripts  # SO:UK
cd $rundir

bb_awg_scripts_dir=/global/homes/k/kwolz/bbdev/bb-awg-scripts  # NERSC
# bb_awg_scripts_dir=/shared_home/kwolz/bbdev/bb-awg-scripts  # SO:UK

export OMP_NUM_THREADS=1
module load soconda

config=/global/homes/k/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/configs/config_deproj_kx40_20arcmin_ndep2000.yaml

srun -n 64 -c 8 --cpu_bind=cores \
python -u ${bb_awg_scripts_dir}/pipeline/simpure/scripts/iterative_purification.py --config $config \
> "${log}" 2>&1
