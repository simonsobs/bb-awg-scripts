#! /bin/bash

. setup.sh

srun -n 224 -c 2 --cpu_bind=cores \
python ${bb_awg_scripts_dir}/pipeline/filtering/filter_simple.py \
    --config_file $filtering_config
