#! /bin/bash

# . setup_cmb_sims.sh  # for CMB sims
. setup_tf_sims.sh


srun -n 30 -c 14 --cpu-bind=cores \
python ${bb_awg_scripts_dir}/pipeline/filtering/coadd_simple.py \
    --config_file $filtering_config
