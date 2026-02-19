#!/bin/bash -l

set -e

# General 
instruments=("satp1")  # ("satp1" "satp3")
sat_freqs=("f090" "f150")

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

        paramfile="paramfile_${patch}_${split}_filtered_${sat}_${sat_freqs}.yaml"
        
        echo "Using paramfile: $paramfile"

        # Uncomment the steps you want to run
        # # This following script was taken from soopercool's "sa_dev" branch
        # python -u ${soopercool_dir}/pipeline/get_analysis_mask.py --globals ${paramfile}
        # python -u ${soopercool_dir}/pipeline/get_mode_coupling.py --globals ${paramfile}
        # python -u ${soopercool_dir}/pipeline/get_full_couplings.py --globals ${paramfile}
        # python -u ${soopercool_dir}/pipeline/compute_pseudo_cells.py --globals ${paramfile} --verbose
        python -u ${soopercool_dir}/pipeline/coadd_pseudo_cells.py --globals ${paramfile} --no_plots
        echo "=== Done with sat: $sat | split: $split | filtered as: $sat $sat_freqs ==="
        echo
    done
  done
done