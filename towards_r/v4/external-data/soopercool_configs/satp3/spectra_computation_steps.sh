#!/bin/bash -l

set -e

# General 
instruments=("satp3")  # ("satp1" "satp3")
sat_freqs=("f090")  #("f090" "f150")

# For now, we only consider south patch.
patches=("south")

soopercool_dir=/home/ar3186/SOOPERCOOL # Insert path to your local SOOPERCOOL directory

# You can add more/take away splits here
splits=("science")

# Run each step for each instrument and split
for sat in "${instruments[@]}"; do
  for sat_freq in "${sat_freqs[@]}"; do
    for patch in "${patches[@]}"; do
      for split in "${splits[@]}"; do
          echo "=== Running for sat: $sat | split: $split ==="

          paramfile="paramfile_${patch}_${split}_filtered_${sat}_${sat_freq}.yaml"
          
          echo "Using paramfile: $paramfile"

          # Uncomment the steps you want to run
          # # This following script was taken from soopercool's "sa_dev" branch
          # srun -n 1 -c 112 --cpu_bind=cores python ${soopercool_dir}/pipeline/get_analysis_mask.py --globals ${paramfile}
          # srun -n 1 -c 112 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/get_mode_coupling.py --globals ${paramfile}
          # srun -n 12 -c 8 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/transfer/compute_pseudo_cells_tf_estimation.py --globals ${paramfile} 
          # srun -n 1 -c 112 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/transfer/compute_transfer_function.py --globals ${paramfile}
          # srun -n 1 -c 112 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/get_full_couplings.py --globals ${paramfile}
          # srun -n 1 -c 112 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/compute_pseudo_cells.py --globals ${paramfile} --verbose
          # srun -n 1 -c 112 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/coadd_pseudo_cells.py --globals ${paramfile} --no_plots
          # srun -n 12 -c 8 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/precompute_cov_couplings.py --globals ${paramfile}
          # srun -n 1 -c 112 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/prepare_cov_inputs.py --globals ${paramfile}
          # srun -n 112 -c 4 --cpu_bind=cores python -u ${soopercool_dir}/pipeline/compute_covariance.py --globals ${paramfile}
          # srun -n 1 -c 112 python -u ${soopercool_dir}/pipeline/create_sacc_file_analytic.py --globals ${paramfile} --data
          srun -n 1 -c 112 python ${soopercool_dir}/pipeline/postproc/fit_cals_and_beams.py --sacc-file /scratch/gpfs/SIMONSOBS/external_data_e2e/v4/spectra/filtered_${sat}_${sat_freq}_20260220/south/full-dataset/science/saccs/cl_and_cov_sacc.fits --lmin-fit 150 --lmax-fit 600 --map-sets-ref planck_f100_filtered_${sat}_${sat_freq}_south_science,planck_f143_filtered_${sat}_${sat_freq}_south_science --map-sets-to-fit ${sat}_f090_south_science,${sat}_f150_south_science --lmax-sacc 650
          echo "=== Done with sat: $sat | split: $split | filtered as: $sat $sat_freq ==="
      done
    done
  done
done