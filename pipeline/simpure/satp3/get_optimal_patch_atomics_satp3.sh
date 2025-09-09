# # Make list
atomic_db=/cephfs/soukdata/sat_analysis/iso_data/mapmaking/satp3_20250108/atomic_db.sqlite  ## YOUR INPUT ATOMIC DB
car_map_template=../band_car_fejer1_5arcmin.fits
output_dir=/cephfs/soukdata/user_data/kwolz/simpure/sky_patch/satp3_iso  # YOUR OUTPUT DIR
freq_channel=f090
delta_ra=1
delta_dec=1

## Specs for limiting obs to a RA/DEC boxed region.
radec_minmax="[-180,60,-10,0]"  # small (fsky 0.04 / natomics ~80)
# radec_minmax="[-180,180,-20,20]"  # large (fsky 0.1 / nsatomics ~210)

mkdir -p $output_dir

## This makes a list of atomic maps based on obs-level restrictions, such as:
## * delta_dec: minimum DEC distance between neighboring maps
## * delta_ra: minimum RA distance between neighboring maps
## * radec_minmax: sky box in degrees, format: [RA_min,RA_max,DEC_min,DEC_max]
python ../get_atomics_list.py \
    --atomic_db $atomic_db \
    --output_dir $output_dir \
    --delta_dec $delta_dec \
    --delta_ra $delta_ra \
    --freq_channel $freq_channel \
    --radec_minmax ${radec_minmax}

map_string_format="satp3_{name_tag}_map.fits"
null_prop_val=science
bundle_db=/pscratch/sd/k/kwolz/bbdev/simpure/sky_patch/single_bundle.db
atomic_list="${output_dir}/atomic_list.npz"
echo $atomic_list

## This bundles the listed atomics with the required specifications and 
## saves them under "{output_dir}/{map_string_format}"
python ../../../pipeline/bundling/bundle_maps.py \
    --bundle_db ${bundle_db} \
    --atomic_db ${atomic_db} \
    --atomic_list ${atomic_list} \
    --output_dir ${output_dir} \
    --map_string_format ${map_string_format} \
    --freq_channel ${freq_channel} \
    --n_bundles 1 \
    --null_prop_val ${null_prop_val} \
    --pix_type car \
    --car_map_template ${car_map_template}

# # Compute fsky (for documentation/validation purposes only)
python compute_fsky.py \
    --hits_map "${output_dir}/satp3_f090_science_hits.fits" \
    --pix_type car
