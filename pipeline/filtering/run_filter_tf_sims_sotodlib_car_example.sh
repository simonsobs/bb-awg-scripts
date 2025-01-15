null_split=science  # For null splits, replace "science" by "low_pwv" or "high_pwv"
sim_ids=0
freq_channel=f150
bundle_id=0
n_bundles=1
pix_type=car

output_dir=/pscratch/sd/k/kwolz/bbdev/simpure/satp3_20240714_small_fp_thin8
atomic_db=/global/cfs/cdirs/sobs/users/susanna/SO_analysis/cmb_maps/satp3/cmb_maps_satp3_20240714/atomic_maps.db
# atomic_list="/pscratch/sd/k/kwolz/bbdev/simpure/sky_patch/small/atomic_list.npz"
car_map_template=/global/cfs/cdirs/sobs/users/susanna/SO_analysis/cmb_maps/satp3/cmb_maps_satp3_20240714/coadded_maps/map_f090_muKcmb.fits
bundle_db="${output_dir}/bundles_seed3456.db"
preprocess_config=/global/homes/k/kwolz/bbdev/bb-awg-scripts/users/kwolz/configs/minimal_preprocess_satp3.yaml
map_dir="/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/tf_sims"
map_string_formats=("pureB_nside512_fwhm30.0_sim{sim_id:04d}.fits" "pureE_nside512_fwhm30.0_sim{sim_id:04d}.fits" "pureT_nside512_fwhm30.0_sim{sim_id:04d}.fits")


for map_string_format in "${map_string_formats[@]}";
do
    time srun -n 32 -c 32 --cpu_bind=cores python ../../../pipeline/filtering/filter_sims_sotodlib.py \
        --atomic-db ${atomic_db} \
        --bundle-db ${bundle_db} \
        --n_bundles ${n_bundles} \
        --preprocess-config ${preprocess_config}\
        --map-dir ${map_dir} \
        --map_string_format ${map_string_format}\
        --sim-ids ${sim_ids}\
        --output-directory "${output_dir}/tf_sims/filtered_${null_split}" \
        --freq-channel ${freq_channel} \
        --bundle-id ${bundle_id} \
        --null_prop_val_inter_obs ${null_split} \
        --pix_type ${pix_type} \
        --car_map_template ${car_map_template} \
        --fp-thin 8
        # --atomic_list ${atomic_list} \

    time python ../../../pipeline/filtering/coadd_filtered_sims.py \
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