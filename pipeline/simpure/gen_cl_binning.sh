#!/bin/bash

. setup.sh

python ${soopercool_dir}/pipeline/misc/get_binning.py \
    --globals ${soopercool_config} \
    --deltal 15