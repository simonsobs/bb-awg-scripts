#! /bin/bash

. setup.sh

# srun -n 112 -c 4 --cpu-bind=cores \  # fine for filtering, not for coaddition cause more space needed
srun -n 32 -c 14 --cpu-bind=cores \
python ${bb_awg_scripts_dir}/pipeline/filtering/coadd_filtered_ext.py \
    --config_file $filtering_config
