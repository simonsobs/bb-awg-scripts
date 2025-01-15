atomic_db=/global/cfs/cdirs/sobs/users/susanna/SO_analysis/cmb_maps/satp3/cmb_maps_satp3_20240714/atomic_maps.db
out_dir=/pscratch/sd/k/kwolz/bbdev/simpure/satp3_20240714_fp_thin8/maps
bundle_db=/pscratch/sd/k/kwolz/bbdev/simpure/satp3_20240714_fp_thin8/bundling/bundles_seed3456.db
car_map_template=/global/cfs/cdirs/sobs/users/susanna/SO_analysis/cmb_maps/satp3/cmb_maps_satp3_20240714/coadded_maps/map_f090_muKcmb.fits
map_string_format="satp1_{name_tag}_bundle{bundle_id}_map.fits"
null_prop_vals=("science")
n_bundles=1
pix_type=car

for null_prop_val in "${null_prop_vals[@]}";
do
    echo $null_prop_val
    python ../../../pipeline/bundling/bundle_maps.py \
        --bundle_db ${bundle_db} \
        --atomic_db ${atomic_db} \
        --output_dir ${out_dir} \
        --map_string_format ${map_string_format} \
        --freq_channel f150 \
        --n_bundles ${n_bundles} \
        --null_prop_val ${null_prop_val} \
        --pix_type ${pix_type} \
        --car_map_template ${car_map_template} \
        --seed 3456 \
        --overwrite
done
