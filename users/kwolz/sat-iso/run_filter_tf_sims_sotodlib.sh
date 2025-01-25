null_split=science  # For null splits, replace "science" by "low_pwv" or "high_pwv"
sim_ids=0
freq_channel=f150
bundle_id=0
pix_type=car

output_dir=/pscratch/sd/k/kwolz/bbdev/sat-iso/satp3_20250108_v0_obs_7
atomic_db=/global/cfs/cdirs/sobs/sat-iso/mapmaking/satp3_20250108_v0/atomic_db.sqlite
atomic_list="/pscratch/sd/k/kwolz/bbdev/sat-iso/satp3_20250108_v0/atomic_list_obs_7.npz"
car_map_template=/pscratch/sd/k/kwolz/bbdev/simpure/sky_patch/fullsky_geometry.fits
bundle_db=/pscratch/sd/k/kwolz/bbdev/sat-iso/satp3_20250108_v0/bundling/bundles_seed3456.db
#preprocess_config=/global/homes/k/kwolz/bbdev/bb-awg-scripts/users/kwolz/configs/minimal_preprocess_susanna_satp3.yaml
preprocess_config_init=/global/homes/k/kwolz/bbdev/bb-awg-scripts/users/kwolz/configs/preprocessing_config_20250108_sat-iso_init.yaml
preprocess_config_proc=/global/homes/k/kwolz/bbdev/bb-awg-scripts/users/kwolz/configs/preprocessing_config_20250108_sat-iso_proc.yaml
map_dir=/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/tf_sims
map_string_formats=("pureT_fwhm30.0_sim{sim_id:04d}_CAR.fits")  # "pureE_fwhm30.0_sim{sim_id:04d}_CAR.fits" "pureB_fwhm30.0_sim{sim_id:04d}_CAR.fits")

mkdir -p $output_dir


for map_string_format in "${map_string_formats[@]}";
do
    echo "${map_string_format}"
    # srun -n 40 -c 4 --cpu_bind=cores 
    # python ../../../pipeline/filtering/filter_sims_sotodlib.py \
    #     --atomic-db ${atomic_db} \
    #     --bundle-db ${bundle_db} \
    #     --preprocess-config-init ${preprocess_config_init} \
    #     --preprocess-config-proc ${preprocess_config_proc} \
    #     --map-dir ${map_dir} \
    #     --map_string_format ${map_string_format}\
    #     --sim-ids ${sim_ids}\
    #     --output-directory "${output_dir}/tf_sims/filtered_${null_split}" \
    #     --freq-channel ${freq_channel} \
    #     --bundle-id ${bundle_id} \
    #     --null_prop_val_inter_obs ${null_split} \
    #     --pix_type ${pix_type} \
    #     --car_map_template ${car_map_template} \
    #     --fp-thin 8 \
    #     --atomic_list ${atomic_list}

    python ../../../pipeline/filtering/coadd_filtered_sims.py \
        --atomic-db ${atomic_db} \
        --bundle-db ${bundle_db} \
        --map_string_format ${map_string_format}\
        --sim-ids ${sim_ids}\
        --output-directory "${output_dir}/tf_sims/filtered_${null_split}" \
        --freq-channel ${freq_channel} \
        --bundle-id ${bundle_id} \
        --null_prop_val_inter_obs ${null_split} \
        --pix_type ${pix_type} \
        --car_template_map ${car_map_template}
done