#!/bin/bash

. setup.sh

python ${soopercool_dir}/pipeline/get_analysis_mask.py \
    --globals ${soopercool_config}
