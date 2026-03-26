#!/bin/bash -l

## Software environment
install_dir=/home/kw6905/bbdev  # YOUR LOCAL BBPower INSTALL PATH
# cd $install_dir; git clone git@github.com:simonsobs/BBPower.git

export OMP_NUM_THREADS=4
export PYTHONPATH="${PYTHONPATH}:${install_dir}/BBPower"

run_dir=/home/kw6905/bbdev/bb-awg-scripts/users/kwolz/bbpower_tutorial
cd $run_dir

mkdir -p out_ISOv1 out_forecast

# Run bbpower on ISO v1 data (SATp3 90+150, Planck 30+100+145+217, WMAP 23)
# Run CL-based component separation
python -u ${install_dir}/BBPower/bbpower/compsep_nopipe.py --config config_ISOv1.yml
# Plot spectra and posteriors 
python -u ${install_dir}/BBPower/bbpower/plotter_nopipe.py --config config_ISOv1.yml

# Run bbpower on simulated CLs (SATs nominal goal-optimistic)
# Generate simulated CLs (CMB+noise+Gaussian FGs) and save to out_forecast
cd "${install_dir}/BBPower"
python -u ${install_dir}/BBPower/examples/generate_SO_spectra.py $run_dir/out_forecast --so_forecast
cd $run_dir
# Run CL-based component separation
python -u ${install_dir}/BBPower/bbpower/compsep_nopipe.py --config config_forecast.yml
# Plot spectra and posteriors 
python -u ${install_dir}/BBPower/bbpower/plotter_nopipe.py --config config_forecast.yml
