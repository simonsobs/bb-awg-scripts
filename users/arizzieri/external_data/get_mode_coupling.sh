#! /bin/bash

unset SLURM_JOB_ID SLURM_NODELIST SLURM_NTASKS SLURM_JOB_NODELIST SLURM_PROCID

. setup.sh

/usr/bin/time -v python ${soopercool_dir}/pipeline/get_mode_coupling.py \
    --globals ${soopercool_config}
