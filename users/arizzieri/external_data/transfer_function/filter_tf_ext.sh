#! /bin/bash

. setup.sh

# srun -n 112 -c 4 --cpu_bind=cores \
python ${bb_awg_scripts_dir}/pipeline/filtering/filter_ext_sotodlib.py \
    --config_file $filtering_config
