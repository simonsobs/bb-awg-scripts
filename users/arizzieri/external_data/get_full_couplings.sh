#! /bin/bash

. setup.sh

python ${soopercool_dir}/pipeline/get_full_couplings.py \
    --globals ${soopercool_config}
