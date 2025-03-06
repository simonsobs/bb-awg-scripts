#! /bin/bash

bb_awg_scripts_dir=/home/kw6905/bbdev/bb-awg-scripts
filtering_config=configs/filtering_config.yaml


srun -n 112 -c 4 --cpu_bind=cores \
python ${bb_awg_scripts_dir}/pipeline/filtering/filter_sims_sotodlib.py \
    --config_file $filtering_config
