#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=4
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=cov-sims

set -e

# Log file
log="./log_cov_sims"

export OMP_NUM_THREADS=1

module use --append /scratch/gpfs/SIMONSOBS/modules
module load soconda

## You will need to git clone SOOPERSIMS repository if not one yet
## https://github.com/simonsobs/SOOPERSIMS/tree/main
soopersims_dir=/home/kw6905/bbdev/SOOPERSIMS  ## YOUR SOOPERSIMS DIRECTORY

paramfile_fit=paramfile_cov_fit_v4.yaml
echo "Running pipeline with paramfile: ${paramfile_fit}"

## Generate Gaussian simulations with CMB and foreground parameters given by
## paramfile_fit.
com="srun -n 100 -c 4 --cpu_bind=cores \
     python -u \
     $soopersims_dir/scripts/cov_sims.py --globals $paramfile_fit"


## NOTE: These sims have already been run.
## Before rerunning it, contact the owner of output_dir (in paramfile_fit)
echo ${com}
echo "Launching pipeline at $(date)"
eval ${com} > ${log} 2>&1
echo "Ending batch script at $(date)"
