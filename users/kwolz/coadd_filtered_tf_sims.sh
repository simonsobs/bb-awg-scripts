#! /bin/bash

bb_awg_scripts_dir=/home/kw6905/bbdev/bb-awg-scripts
filtering_config=configs/filtering_config.yaml


srun -n 32 -c 14 --cpu-bind=cores \
python ${bb_awg_scripts_dir}/pipeline/filtering/coadd_filtered_sims.py \
    --config_file $filtering_config
