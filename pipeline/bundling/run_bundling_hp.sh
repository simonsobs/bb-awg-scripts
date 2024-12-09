atomic_db=/pscratch/sd/r/rosenber/so_pwg/hp_mapmaker/satp1_241031/atomic_maps_satp1_241031.db

out_dir=/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/maps
map_string_format="satp1_{name_tag}_bundle{bundle_id}_map.fits"
null_prop_vals=("low_elevation", "high_elevation", "low_pwv", "high_pwv", "None")

for null_prop_val in "${null_prop_vals[@]}";
do
    echo $null_prop_val
    bundle_db=/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/bundling/bundles_${null_prop_val}.db
    python bundle_maps.py \
        --bundle_db ${bundle_db} \
        --atomic_db ${atomic_db} \
        --output_dir ${out_dir} \
        --map_string_format ${map_string_format} \
        --freq_channel f150 \
        --n_bundles 2 \
        --null_prop_val ${null_prop_val} \
        --pix_type hp
done
    