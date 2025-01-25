atomic_db=/pscratch/sd/r/rosenber/so_pwg/hp_mapmaker/satp1_241031/atomic_maps_satp1_241031.db
out_dir=/pscratch/sd/k/kwolz/bbdev/bb-awg-scripts/tf_tests/end_to_end_2412_fp_thin8/maps
bundle_db=/pscratch/sd/k/kwolz/bbdev/bb-awg-scripts/tf_tests/end_to_end_2412_fp_thin8/bundling/bundles_seed3456.db
map_string_format="satp1_{name_tag}_bundle{bundle_id}_map.fits"
null_prop_vals=("science")

for null_prop_val in "${null_prop_vals[@]}";
do
    echo $null_prop_val
    python ../../../pipeline/bundling/bundle_maps.py \
        --bundle_db ${bundle_db} \
        --atomic_db ${atomic_db} \
        --output_dir ${out_dir} \
        --map_string_format ${map_string_format} \
        --freq_channel f150 \
        --n_bundles 2 \
        --null_prop_val ${null_prop_val} \
        --pix_type hp \
        --seed 3456 \
        --null_props pwv elevation \
        --only_make_db \
        --overwrite
done
