#! /bin/bash

. setup.sh

python ${soopercool_dir}/pipeline/transfer/compute_transfer_function.py --globals ${soopercool_config}
python ${soopercool_dir}/pipeline/get_mode_coupling.py --globals ${soopercool_config}
python ${soopercool_dir}/pipeline/get_full_couplings.py --globals ${soopercool_config}

# TF validation; to run need to pull PR https://github.com/simonsobs/SOOPERCOOL/pull/58
#python ${soopercool_dir}/pipeline/transfer/compute_pseudo_cells_tf_validation.py --globals ${soopercool_config} --verbose
#python ${soopercool_dir}/pipeline/transfer/validate_transfer_function.py --globals ${soopercool_config}