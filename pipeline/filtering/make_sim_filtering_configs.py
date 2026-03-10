import argparse
import numpy as np
import os


def filter_string(rundir, outdir, bb_awg_scripts_dir,
                  id_sim_first, id_sim_last, tel,
                  email):
    """
    This makes a slurm script that is optimised to filtering atomic maps with
    sotodlib on tiger3 using MPI parallelization.
    """
    if id_sim_last == id_sim_first:
        sim_string = str(id_sim_last)
    else:
        sim_string = f"{id_sim_first},{id_sim_last}"
    nnodes = 50
    ntasks = int(112*nnodes)
    string = """#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes={nnodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --job-name={tel}-cov-sims-filter-{sim_string}
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=ALL

set -e

# Log file
log="{outdir}/log_{tel}_cov_sims_filter_{sim_string}"

export OMP_NUM_THREADS=1

rundir={rundir}  ## YOUR RUNNING DIRECTORY
outdir={outdir}  ## YOUR LOGGING DIRECTORY
bb_awg_scripts_dir={bb_awg_scripts_dir}  ## YOUR bb_awg_scripts DIRECTORY

cd $rundir
filtering_config="${{outdir}}/filtering_config_covar_{tel}_full.yaml"

# This produces filtered atomics
NOW=$( date '+%F_%H:%M:%S' )
srun -n {ntasks} -c 1 --cpu_bind=cores \\
python -u \\
    ${{bb_awg_scripts_dir}}/pipeline/filtering/filter_sims_sotodlib.py \\
    --config_file $filtering_config --sim_ids {sim_string} \\
        > "${{log}}_${{NOW}}" 2>&1

echo "Ending batch script at $(date)"
    """.format(rundir=rundir, outdir=outdir,
               bb_awg_scripts_dir=bb_awg_scripts_dir,
               sim_string=sim_string, tel=tel, user_email=email,
               nnodes=nnodes, ntasks=ntasks)
    return string


def coadd_string(rundir, outdir, bb_awg_scripts_dir, id_sim_first, id_sim_last,
                 tel, email):
    """
    This makes a slurm script that is optimised to coadding filtered atomic
    maps with sotodlib on tiger3 usin MPI parallelization, meant to be
    executed after running the filtering script.
    """
    if id_sim_last == id_sim_first:
        sim_string = str(id_sim_last)
    else:
        sim_string = f"{id_sim_first},{id_sim_last}"
    nnodes = 20
    ntasks = int(112*nnodes/14)
    string = """#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes={nnodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task=14
#SBATCH --time=10:00:00
#SBATCH --job-name={tel}-cov-sims-coadd-{sim_string}
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=ALL

set -e

# Log file
log="{outdir}/log_{tel}_cov_sims_coadd_{sim_string}"

export OMP_NUM_THREADS=1

rundir={rundir}  ## YOUR RUNNING DIRECTORY
outdir={outdir}  ## YOUR LOGGING DIRECTORY
bb_awg_scripts_dir={bb_awg_scripts_dir}  ## YOUR bb_awg_scripts DIRECTORY

cd $rundir
filtering_config="${{outdir}}/filtering_config_covar_{tel}_full.yaml"

# This coadds them into bundles
NOW=$( date '+%F_%H:%M:%S' )
srun -n {ntasks} -c 14 --cpu_bind=cores \\
python -u \\
    ${{bb_awg_scripts_dir}}/pipeline/filtering/coadd_filtered_sims.py \\
    --config_file $filtering_config --sim_ids {sim_string} \\
        > "${{log}}_${{NOW}}" 2>&1

echo "Ending batch script at $(date)"
    """.format(rundir=rundir, outdir=outdir,
               bb_awg_scripts_dir=bb_awg_scripts_dir,
               sim_string=sim_string, tel=tel, user_email=email,
               nnodes=nnodes, ntasks=ntasks)
    return string


def delete_string(rundir, outdir, bb_awg_scripts_dir,
                  id_sim_first, id_sim_last, tel, email):
    """
    This makes a slurm script that serially deletes from disk all filtered
    atomic maps, meant to be executed after coadding them into bundles.
    """
    if id_sim_last == id_sim_first:
        sim_string = str(id_sim_last)
    else:
        sim_string = f"{id_sim_first},{id_sim_last}"
    string = """#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:61:00
#SBATCH --job-name={tel}-cov-sims-delete-{sim_string}
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=ALL

set -e

# Log file
log="{outdir}/log_{tel}_cov_sims_delete_{sim_string}"

export OMP_NUM_THREADS=1

module use --append /scratch/gpfs/SIMONSOBS/modules
module load soconda

rundir={rundir}  ## YOUR RUNNING DIRECTORY
outdir={outdir}  ## YOUR LOGGING DIRECTORY
bb_awg_scripts_dir={bb_awg_scripts_dir}  ## YOUR bb_awg_scripts DIRECTORY

cd $rundir
filtering_config="${{outdir}}/filtering_config_covar_{tel}_full.yaml"

# This coadds them into bundles
NOW=$( date '+%F_%H:%M:%S' )
python -u \\
    ${{bb_awg_scripts_dir}}/pipeline/filtering/delete_atomic_sims.py \\
    --config_file $filtering_config --sim_ids {sim_string} --force_delete \\
        > "${{log}}_${{NOW}}" 2>&1

echo "Ending batch script at $(date)"
    """.format(rundir=rundir, outdir=outdir,
               bb_awg_scripts_dir=bb_awg_scripts_dir,
               sim_string=sim_string, tel=tel, user_email=email)
    return string


def main(args):
    """
    This script makes slurm scripts that produce simulated
    filtered bundles in batches of {batch_size} simulations on tiger.
    The resulting scripts are:
    * filter_{id_batch}.sh  # Filter single batch of simulations
    * coadd_{id_batch}.sh  # Coadd resulting atomics in bundles
    * delete_{id_batch}.sh  # Delete atomics after enusring bundles exist

    Args:
        rundir: str
            Running directory where the ISO processing scripts live.
        outdir: str
            Output directory for sim-filtering scripts.
        bb_awg_scripts_dir:
            Install directory of bb-awg-scripts repository.
            See https://github.com/simonsobs/bb-awg-scripts
        batch_size: int
            Number of filtered simulations to be generated in a single batch.
        id_sim_first: int
            First simulation ID to be filtered
        id_sim_last: int
            Last simulation ID to be filtered
        tel: str
            Telescope. Choose between 'satp1' and 'satp3'.
        email: str
            User email for SLURM notifications about job failure.
    """
    outdir = os.path.abspath(args.outdir)
    rundir = os.path.abspath(args.rundir)
    bb_awg_scripts_dir = os.path.abspath(args.bb_awg_scripts_dir)
    id_sim_first = args.id_sim_first
    id_sim_last = args.id_sim_last
    num_sims = id_sim_last - id_sim_first + 1
    batch_size = args.batch_size
    num_batches = int(np.ceil(float(num_sims)/float(batch_size)))
    email = args.email

    import glob
    for typ in ["filter", "coadd", "delete"]:
        for f in glob.glob(f"{outdir}/{typ}_*.sh"):
            os.remove(f)

    for id_batch in range(num_batches):
        idfirst = id_sim_first + id_batch*batch_size
        idlast = min(id_sim_first + (id_batch+1)*batch_size - 1, id_sim_last)
        print("id_batch", id_batch, idfirst, idlast)

        print(f"{outdir}/filter_{id_batch}.sh",)

        with open(f"{outdir}/filter_{id_batch}.sh", "w") as f:
            f.write(filter_string(rundir, outdir, bb_awg_scripts_dir,
                                  idfirst, idlast,
                                  args.tel, email))
        with open(f"{outdir}/coadd_{id_batch}.sh", "w") as f:
            f.write(coadd_string(rundir, outdir, bb_awg_scripts_dir,
                                 idfirst, idlast,
                                 args.tel, email))
        with open(f"{outdir}/delete_{id_batch}.sh", "w") as f:
            f.write(delete_string(rundir, outdir, bb_awg_scripts_dir,
                                  idfirst, idlast,
                                  args.tel, email))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rundir", type=str)
    parser.add_argument("--outdir", type=str, default=os.getcwd())
    parser.add_argument("--bb_awg_scripts_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--id_sim_first", type=int)
    parser.add_argument("--id_sim_last", type=int)
    parser.add_argument("--tel", type=str)
    parser.add_argument("--email", type=str)

    args = parser.parse_args()
    main(args)
