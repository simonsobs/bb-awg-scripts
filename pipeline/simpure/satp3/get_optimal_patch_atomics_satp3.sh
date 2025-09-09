# # Make list
atomic_db=/cephfs/soukdata/sat_analysis/iso_data/mapmaking/satp3_20250108/atomic_db.sqlite  # /scratch/gpfs/SIMONSOBS/sat-iso/mapmaking/satp3_20250108/atomic_db.sqlite
# car_map_template=/scratch/gpfs/SIMONSOBS/sat-iso/transfer_function/soopercool_inputs/band_car_fejer1_5arcmin.fits
output_dir=/cephfs/soukdata/user_data/kwolz/simpure/sky_patch/satp3_iso  # /scratch/gpfs/SIMONSOBS/users/kw6905/simpure/sky_patch/satp3_iso
freq_channel=f090
delta_ra=1
delta_dec=1
radec_minmax="[-180,60,-10,0]"  # small (fsky 0.04 / natomics ~80)
# radec_minmax="[-180,180,-20,20]"  # large (fsky 0.1 / nsatomics ~210)

mkdir -p $output_dir

python ../get_atomics_list.py \
    --atomic_db $atomic_db \
    --output_dir $output_dir \
    --delta_dec $delta_dec \
    --delta_ra $delta_ra \
    --freq_channel $freq_channel \
    --radec_minmax ${radec_minmax}

# # Bundle atomics
# map_string_format="satp3_{name_tag}_map.fits"
# null_prop_val=science
# bundle_db=/pscratch/sd/k/kwolz/bbdev/simpure/sky_patch/single_bundle.db
# atomic_list="${output_dir}/atomic_list.npz"
# echo $atomic_list

# python ../../../pipeline/bundling/bundle_maps.py \
#     --bundle_db ${bundle_db} \
#     --atomic_db ${atomic_db} \
#     --atomic_list ${atomic_list} \
#     --output_dir ${output_dir} \
#     --map_string_format ${map_string_format} \
#     --freq_channel ${freq_channel} \
#     --n_bundles 1 \
#     --null_prop_val ${null_prop_val} \
#     --pix_type car \
#     --car_map_template ${car_map_template}

# # Compute fsky
# python compute_fsky.py \
#     --hits_map "${output_dir}/satp3_f090_science_hits.fits" \
#     --pix_type car
