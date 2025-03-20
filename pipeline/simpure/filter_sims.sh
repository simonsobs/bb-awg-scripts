#! /bin/bash

. setup.sh

srun -n 112 -c 4 --cpu_bind=cores \
python ${bb_awg_scripts_dir}/pipeline/filtering/filter_sims_simpure.py \
    --config_file $filtering_config
