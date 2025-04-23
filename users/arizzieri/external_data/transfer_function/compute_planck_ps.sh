#! /bin/bash

. setup.sh

python ${soopercool_dir}/pipeline/compute_pseudo_cells.py \
    --globals ${soopercool_config} --no_plots
