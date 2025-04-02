#! /bin/bash

. setup.sh

# Must set n = (# sims) * (# map sets) * (# map sets + 1) / 2
# maximum allowed c = (# nodes) * 112 / n  (at tiger3)
# srun -n 5 -c 4 --cpu-bind=cores \
python ${soopercool_dir}/pipeline/transfer/compute_pseudo_cells_tf_estimation.py \
    --globals ${soopercool_config}
