#! /bin/bash

. setup.sh

srun -n 15 -c 14 --cpu-bind=cores \
python ${bb_awg_scripts_dir}/pipeline/filtering/coadd_simple.py \
    --config_file $filtering_config
