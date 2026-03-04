#!/bin/bash -l

set -e

# General 
instruments=("satp1")  # ("satp1" "satp3")
sat_freqs=("f090")  # "f150")

# For now, we only consider south patch.
patches=("south")

export OMP_NUM_THREADS=1

soopercool_dir=/home/ar3186/SOOPERCOOL # Insert path to your local SOOPERCOOL directory

# You can add more/take away splits here
splits=("science")

# Run each step for each instrument and split
for sat in "${instruments[@]}"; do
  for patch in "${patches[@]}"; do
    for split in "${splits[@]}"; do
        echo "=== Running for sat: $sat | split: $split ==="

        paramfile="paramfile_${patch}_${split}_unfiltered_spass.yaml"
        
        echo "Using paramfile: $paramfile"

        # Uncomment the steps you want to run
        # This following script was taken from soopercool's "sa_dev" branch
        # srun -n 1 -c 112 python -u ${soopercool_dir}/pipeline/get_mode_coupling.py --globals ${paramfile}
        # echo "=== Done with get_mode_coupling.py ==="
        # srun -n 1 -c 112 python -u ${soopercool_dir}/pipeline/get_full_couplings.py --globals ${paramfile}
        # echo "=== Done with get_full_couplings.py ==="
        srun -n 1 -c 112 python -u ${soopercool_dir}/pipeline/compute_pseudo_cells.py --globals ${paramfile} --verbose
        # echo "=== Done with compute_pseudo_cells.py ==="
        # srun -n 1 -c 112 python -u ${soopercool_dir}/pipeline/coadd_pseudo_cells.py --globals ${paramfile}
        # srun -n 1 -c 112 python -u ${soopercool_dir}/pipeline/prepare_cov_inputs.py --globals ${paramfile}
        # srun -n 16 -c 28 python -u ${soopercool_dir}/pipeline/precompute_cov_couplings.py --globals ${paramfile}
        # srun -n 16 -c 28 python -u ${soopercool_dir}/pipeline/compute_covariance.py --globals ${paramfile}
        # srun -n 1 -c 1 python -u ${soopercool_dir}/pipeline/create_sacc_file.py

        echo "=== Done with sat: $sat | split: $split | filtered as: $sat $sat_freqs ==="
        echo
    done
  done
done