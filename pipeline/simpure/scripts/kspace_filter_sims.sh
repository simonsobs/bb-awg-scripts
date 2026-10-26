#!/bin/bash

bb_awg_scripts_dir=/global/homes/k/kwolz/bbdev/bb-awg-scripts

set OMP_NUM_THREADS=1

## Generate a set of pure-E simulations needed for E-to-B leakage deprojection
srun -n 112 -c 1 --cpu_bind=cores \
python ${bb_awg_scripts_dir}/pipeline/simpure/scripts/kspace_filter_sims.py