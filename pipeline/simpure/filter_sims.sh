#! /bin/bash

set -e

#. setup_cmb_sims.sh  # for CMB sims
 . setup_tf_sims.sh

# Log file
log="./log_simpure_filtering"

#=========== Run it ============
module use /cephfs/soukdata/software/modulefiles/
module load soconda/20241017_3.10

echo "Launching pipeline at $(date)"

# This code took 186 minutes on 4 tiger nodes and 100 sims
# SATp3: 117, SATp1: 138
srun -n 117 -c 3 --cpu_bind=cores \
python -u ${bb_awg_scripts_dir}/pipeline/filtering/filter_simple.py \
    --config_file $filtering_config \
> ${log} 2>&1

srun -n 30 -c 14 --cpu-bind=cores \
python ${bb_awg_scripts_dir}/pipeline/filtering/coadd_simple.py \
    --config_file $filtering_config \
> ${log} 2>&1

echo "Ending pipeline at $(date)"
