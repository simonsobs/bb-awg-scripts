#! /bin/bash

# . setup_cmb_sims.sh  # for CMB sims
. setup_tf_sims.sh

# Log file
log="./log_simpure_coadding"

#=========== Run it ============
module use /cephfs/soukdata/software/modulefiles/
module load soconda/20241017_3.10

echo "Launching pipeline at $(date)"

srun -n 30 -c 14 --cpu-bind=cores \
python ${bb_awg_scripts_dir}/pipeline/filtering/coadd_simple.py \
    --config_file $filtering_config \
> ${log} 2>&1

echo "Ending pipeline at $(date)"
