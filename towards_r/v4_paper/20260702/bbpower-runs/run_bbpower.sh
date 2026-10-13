#!/bin/bash -l

set -e

basedir=/home/ar3186/awg-clean/bb-awg-scripts/towards_r/v4_paper/20260702/bbpower-runs/  ## YOUR RUNNING DIRECTORY
cd $basedir

bbpower_dir=/home/ar3186/BBPower/  ## PATH TO YOUR LOCAL BBPOWER

bbpower_config=/home/ar3186/awg-clean/bb-awg-scripts/towards_r/v4_paper/20260702/bbpower-runs/sat_data_from_calibrated_maps/paramfile_south_science__satp1__satp3__ext_data_filtered_satp3_f090__gal060_SMICAblob__Fornax_radius_1deg_apo_2deg.yml

echo "Launching pipeline at $(date)"

srun -n 1 -c 112 \
    python -u ${bbpower_dir}/bbpower/compsep_nopipe.py --config $bbpower_config

echo "Finished compsep at $(date)"

srun -n 1 -c 112 \
    python -u ${bbpower_dir}/bbpower/plotter_nopipe.py --config $bbpower_config

echo "Ending batch script at $(date)"
