#! /bin/bash

#. setup.sh
. setup_cmb_sims.sh

# SATp3: 117, SATp1: 138
srun -n 117 -c 3 --cpu_bind=cores \
python ${bb_awg_scripts_dir}/pipeline/filtering/filter_simple.py \
    --config_file $filtering_config
