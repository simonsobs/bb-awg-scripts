null_split=science  # For null splits, replace "science" by "low_pwv" or "high_pwv"
sim_ids=0,4  # A stable option is to loop over 5 sims and 2 bundles for each run
freq_channel="f150"
bundle_ids=(0 1)
nside=512

atomic_db="/pscratch/sd/r/rosenber/so_pwg/hp_mapmaker/satp1_241031/atomic_maps_satp1_241031.db"
bundle_db="/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/bundling/bundles_${null_split}.db"
preprocess_config="/global/cfs/cdirs/sobs/users/rosenber/preprocess/satp1_241031/minimal_preprocess.yaml"
map_dir="/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/cmb_sims"
map_template="cmb_nside512_fwhm30.0_sim{sim_id:04d}.fits"

output_directory=/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/cmb_sims/filtered_${null_split}


for bundle_id in "${bundle_ids[@]}";
do
    time srun -n 64 -c 16 --cpu_bind=cores python filter_sims_sotodlib.py \
        --atomic-db ${atomic_db} \
        --bundle-db ${bundle_db} \
        --preprocess-config ${preprocess_config}\
        --map-dir ${map_dir} \
        --map-template ${map_template}\
        --sim-ids ${sim_ids}\
        --output-directory ${output_directory} \
        --freq-channel ${freq_channel} \
        --bundle-id ${bundle_id} \
        --nside ${nside} \
        --null-prop ${null_split}

    time python coadd_filtered_sims.py \
        --atomic-db ${atomic_db} \
        --bundle-db ${bundle_db} \
        --map-template ${map_template}\
        --sim-ids ${sim_ids}\
        --output-directory ${output_directory} \
        --freq-channel ${freq_channel} \
        --bundle-id ${bundle_id} \
        --nside ${nside} \
        --null-prop ${null_split}
done