# # Make list
atomic_db_in=/cephfs/soukdata/sat_analysis/iso_data/v1/mapmaking/satp3_20250108/atomic_db.sqlite  ## YOUR INPUT ATOMIC DB
car_map_template=band_car_fejer1_5arcmin.fits
output_dir=/cephfs/soukdata/user_data/kwolz/simpure/sky_patch/v1_20251016  # YOUR OUTPUT DIR
atomic_db_out="${output_dir}/atomic_db_ws0.sqlite"
freq_channel=f090
delta_ra=0.5  # 2 # 1
delta_dec=0.5  # 2  # 1

## Specs for limiting obs to a RA/DEC boxed region.
#radec_minmax="[-180,60,-10,0]"  # small (fsky 0.04 / natomics ~80)
#radec_minmax="[-180,180,-20,20]"  # large (fsky 0.1 / natomics ~210)
radec_minmax="[-180,180,-50,50]"  # very large (fsky >0.1 / natomics ~875)

mkdir -p $output_dir

rundir=/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure
cd $rundir 

## This creates a new atomic DB based on obs-level restrictions, such as:
## * delta_dec: minimum DEC distance between neighboring maps
## * delta_ra: minimum RA distance between neighboring maps
## * radec_minmax: sky box in degrees, format: [RA_min,RA_max,DEC_min,DEC_max]
python scripts/get_atomics_list.py \
    --atomic_db $atomic_db_in \
    --output_dir $output_dir \
    --delta_dec $delta_dec \
    --delta_ra $delta_ra \
    --freq_channel $freq_channel \
    --radec_minmax ${radec_minmax}
