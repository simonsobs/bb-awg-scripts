atomic_db=/pscratch/sd/r/rosenber/so_pwg/hp_mapmaker/satp1_241031/atomic_maps_satp1_241031.db

out_dir=/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/signflip_maps
map_string_format="satp1_{name_tag}_bundle{bundle_id}_sim{sim_id:04}_noise.fits"
null_prop_vals=("low_pwv" "high_pwv" "science")

for null_prop_val in "${null_prop_vals[@]}";
do
    echo $null_prop_val
    bundle_db=/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/bundling/bundles_${null_prop_val}.db
    # time srun -n 16 -c 16 --cpu_bind=cores python make_signflip_noise.py \
    time python make_signflip_noise.py \
        --bundle_db ${bundle_db} \
        --atomic_db ${atomic_db} \
        --output_dir ${out_dir} \
        --map_string_format ${map_string_format} \
        --freq_channel f150 \
        --n_sims 100 \
        --n_bundles 2 \
        --null_prop_val ${null_prop_val} \
        --pix_type hp
done
