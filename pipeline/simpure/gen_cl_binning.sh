#!/bin/bash

. setup_tf_sims.sh

python ${soopercool_dir}/pipeline/misc/get_binning.py \
    --globals ${soopercool_config} \
    --deltal 10