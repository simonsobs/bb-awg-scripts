#null_split=high_pwv
sim_ids=3
freq_channel="f150"
bundle_id=0
nside=512

atomic_db="/pscratch/sd/r/rosenber/so_pwg/hp_mapmaker/satp1_241031/atomic_maps_satp1_241031.db"
#bundle_db="/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/bundling/bundles_${null_split}.db"
bundle_db="/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/bundling/bundles_science.db"
preprocess_config="/global/cfs/cdirs/sobs/users/rosenber/preprocess/satp1_241031/minimal_preprocess.yaml"
map_dir="/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/tf_sims"
map_templates=("pureB_nside512_fwhm30.0_sim{sim_id:04d}.fits")

#output_directory=/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/tf_sims/filtered_${null_split}
output_directory=/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/tf_sims/filtered_science


for map_template in "${map_templates[@]}";
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
        --nside ${nside}
        #--null-prop ${null_split}

    time python coadd_filtered_sims.py \
        --atomic-db ${atomic_db} \
        --bundle-db ${bundle_db} \
        --map-template ${map_template}\
        --sim-ids ${sim_ids}\
        --output-directory ${output_directory} \
        --freq-channel ${freq_channel} \
        --bundle-id ${bundle_id} \
        --nside ${nside}
        #--null-prop ${null_split}
done