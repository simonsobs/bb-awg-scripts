atomic_db=/pscratch/sd/k/kwolz/share/e2e_2412/satp1_241031_er/atomic_maps_satp1_241031.db

out_dir=/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/maps
map_string_format="satp1_{name_tag}_bundle{bundle_id}_map.fits"
split_labels_intra_obs=("wafer_low" "wafer_high")

for split_label_intra_obs in "${split_labels_intra_obs[@]}";
do
    echo $split_label_intra_obs
    bundle_db=/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/bundling/bundles_${split_label_intra_obs}.db
    python ../../pipeline/bundling/bundle_maps.py \
        --bundle_db ${bundle_db} \
        --atomic_db ${atomic_db} \
        --output_dir ${out_dir} \
        --map_string_format ${map_string_format} \
        --freq_channel f150 \
        --n_bundles 2 \
        --split_label_intra_obs ${split_label_intra_obs} \
        --pix_type hp
done