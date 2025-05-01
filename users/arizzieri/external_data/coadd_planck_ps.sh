#! /bin/bash

. setup.sh

python ${soopercool_dir}/pipeline/coadd_pseudo_cells.py \
    --globals ${soopercool_config}
