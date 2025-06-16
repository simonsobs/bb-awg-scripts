#! /bin/bash

unset SLURM_JOB_ID SLURM_NODELIST SLURM_NTASKS SLURM_JOB_NODELIST SLURM_PROCID

. setup.sh

python ${soopercool_dir}/pipeline/get_full_couplings.py \
    --globals ${soopercool_config}
