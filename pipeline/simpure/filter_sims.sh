#! /bin/bash

# . setup_cmb_sims.sh  # for CMB sims
. setup_tf_sims.sh

# This code took 186 minutes on 4 tiger nodes and 100 sims
# SATp3: 117, SATp1: 138
srun -n 117 -c 3 --cpu_bind=cores \
python ${bb_awg_scripts_dir}/pipeline/filtering/filter_simple.py \
    --config_file $filtering_config
