#!/bin/bash -l

set -e

soopercool_dir=/home/ar3186/SOOPERCOOL_tests/SOOPERCOOL/ # Insert path to your local SOOPERCOOL directory

paramfile="/home/ar3186/awg-clean/bb-awg-scripts/towards_r/v4_paper/20260702/soopercool-configs/sat_data_from_calibrated_maps/satp1_satp3/paramfile_south_science__satp1__satp3__gal060_SMICAblob__Fornax_radius_1deg_apo_2deg.yaml"

echo "=== Start running ==="
echo "Using paramfile: $paramfile"

# Uncomment the steps you want to run

# Mask
srun -n 1 -c 112 --cpu_bind=cores python ${soopercool_dir}/pipeline/get_analysis_mask.py --globals ${paramfile}
echo "=== Done with get_analysis_mask.py ==="

echo "=== Done ==="
