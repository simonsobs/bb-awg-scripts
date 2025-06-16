#! /bin/bash

unset SLURM_JOB_ID SLURM_NODELIST SLURM_NTASKS SLURM_JOB_NODELIST SLURM_PROCID

. setup.sh

python ${soopercool_dir}/pipeline/coadd_pseudo_cells.py \
    --globals ${soopercool_config}
