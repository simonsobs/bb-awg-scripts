#!/bin/bash

soopercool_dir=/shared_home/kwolz/bbdev/SOOPERCOOL  # YOUR SOOPERCOOL DIR

python ${soopercool_dir}/pipeline/misc/get_binning.py \
    --globals ${soopercool_config} \
    --deltal 10