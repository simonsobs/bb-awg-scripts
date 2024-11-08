#!/bin/bash
## Call as "bash run_mapmaker.sh $scripts_directory $config_file $nproc"
## Make atomic maps, copy configs to output directory, make atomic db, change permissions
## This is as for Perlmutter; you may need to change the software stack or the permissions line on other systems

# Make a function to read entries from config file
load_from_config () {
    local line=$(grep $1: $config) # Get the line containing the variable
    var=${line#$1:*[\'\"]} # Strip off the start
    var=${var%[\'\"]*} # Strip off the end
    if [ -z $var ]; then
	echo "Failed to parse $1 from config file $config"
	exit 1
    fi
}

# Load command line args
workdir=$1  # Directory containing python scripts
config=$2 # Config file should be the second argument
if [ -z $3 ]; then
    nproc="";
else nproc=" --nproc $3"
fi
if [ -z $OMP_NUM_THREADS ]; then
    export OMP_NUM_THREADS=4
fi
    
# Load software environment
module use /global/common/software/sobs/perlmutter/modulefiles
module load soconda

# Go to config directory so relative paths work in config file
configdir=$(dirname ${config})
pwd=$PWD
cd $configdir

# Make output directory
load_from_config odir
odir=$var
mkdir -p $odir

# Make the maps
python $workdir/make_atomic_filterbin_map.py --config $config $nproc

# Copy the config file to the output directory
cp $config $odir

# Make atomic db
load_from_config atomic_db
python $workdir/make_atomic_db.py $odir $var

# Copy preprocess config to the output directory
load_from_config preprocess_config
cp $var $odir

# Change permissions
chgrp -R sobs $odir

cd $pwd
