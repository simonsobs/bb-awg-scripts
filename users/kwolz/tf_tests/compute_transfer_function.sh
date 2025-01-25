. setup_soopercool.sh
python  ~/bbdev/SOOPERCOOL/pipeline/transfer/compute_transfer_function.py --globals ${paramfile}
python  ~/bbdev/SOOPERCOOL/pipeline/get_full_couplings.py --globals ${paramfile}