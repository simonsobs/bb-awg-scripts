#! /bin/bash

. setup.sh

# srun -n 32 -c 14 --cpu-bind=cores \
python ${bb_awg_scripts_dir}/pipeline/filtering/coadd_filtered_sims.py \
    --config_file $filtering_config
