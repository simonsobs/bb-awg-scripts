#!/bin/bash -l

set -e

soopercool_dir=/home/ar3186/SOOPERCOOL_tests/SOOPERCOOL/ # Insert path to your local SOOPERCOOL directory

filtering_sat='satp3_f090'  # satp1_f090 or satp1_f150 or satp3_f090 or satp3_f150

paramfile="/home/ar3186/awg-clean/bb-awg-scripts/towards_r/v4_paper/20260702/soopercool-configs/sat_data_from_calibrated_maps/satp1_satp3_extdata/paramfile_south_science__satp1__satp3__ext_data_filtered_${filtering_sat}__gal060_SMICAblob__Fornax_radius_1deg_apo_2deg.yaml"

path_save_saccs="/scratch/gpfs/SIMONSOBS/external_data_e2e/v4_paper/spectra_20260702/south/science/sat_data_from_calibrated_maps/satp1__satp3__ext_data_filtered_satp3_f090/analysis_mask_satp1_satp3__gal060_SMICAblob__Fornax_radius_1deg_apo_2deg/saccs/"


echo "=== Start running ==="
echo "Using paramfile: $paramfile"

# Uncomment the steps you want to run

# # Kspace sims
# srun -n 40 -c 10 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/kspace/filter_sims_kspace.py --globals ${paramfile} 
# echo "=== Done with kspacesims.py ==="

# Mode coupling
srun -n 1 -c 112 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/get_mode_coupling.py --globals ${paramfile}
echo "=== Done with get_mode_coupling.py ==="

# Transfer function  # NOTE: only run if I set new transfer function dir
srun -n 40 -c 10 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/transfer/compute_pseudo_cells_tf_estimation.py --globals ${paramfile} 
echo "=== Done with compute_pseudo_cells_tf_estimation.py ==="
srun -n 1 -c 112 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/transfer/compute_transfer_function.py --globals ${paramfile}
echo "=== Done with compute_transfer_function.py ==="

# Pseudo-cells
srun -n 1 -c 112 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/compute_pseudo_cells.py --globals ${paramfile} --verbose
echo "=== Done with compute_pseudo_cells.py ==="
srun -n 1 -c 112 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/coadd_pseudo_cells.py --globals ${paramfile} --no_plots
echo "=== Done with coadd_pseudo_cells.py ==="

# Covariance
srun -n 1 -c 112 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/prepare_cov_inputs.py --globals ${paramfile}
echo "=== Done with prepare_cov_inputs.py ==="
srun -n 64 -c 7 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/precompute_cov_couplings.py --globals ${paramfile}
echo "=== Done with precompute_cov_couplings.py ==="
srun -n 112 -c 4 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/compute_covariance.py --globals ${paramfile}
echo "=== Done with compute_covariance.py ==="

# SACC
srun -n 1 -c 112 python -u ${soopercool_dir}/pipeline/create_sacc_file_analytic.py --globals ${paramfile} --data
echo "=== Done with create_sacc_file_analytic.py ==="


# srun -n 1 -c 112 python ${soopercool_dir}/pipeline/postproc/fit_EB_angles.py --sacc-file $path_save_saccs/cl_and_cov_sacc.fits \
#               --lmin-fit 150 --lmax-fit 600 --map-sets-to-fit satp1_f090_south_science,satp1_f150_south_science,satp3_f090_south_science,satp3_f150_south_science --lmax-sacc 650
# echo "=== Done with fit_EB_angles.py ==="

# srun -n 1 -c 112 python ${soopercool_dir}/pipeline/postproc/fit_cals_and_beams.py --sacc-file $path_save_saccs/cl_and_cov_sacc_rotated.fits --lmin-fit 150 --lmax-fit 600 --map-sets-ref planck_f100_filtered_${filtering_sat}_south_science,planck_f143_filtered_${filtering_sat}_south_science,planck_f100_filtered_${filtering_sat}_south_science,planck_f143_filtered_${filtering_sat}_south_science --map-sets-to-fit satp1_f090_south_science,satp1_f150_south_science,satp3_f090_south_science,satp3_f150_south_science --lmax-sacc 650 --no-beam-fit
# echo "=== Done with fit_cals_and_beams.py ==="


echo "=== Done ==="
