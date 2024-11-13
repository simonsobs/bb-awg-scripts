bundle_db="test_null.db"
atomic_db="/pscratch/sd/r/rosenber/so_pwg/hp_mapmaker/satp1_241031/atomic_maps_satp1_241031.db"

freq_channel="f150"
out_dir="noise_maps_241113"

python generate_signflip_noise.py \
    --bundle-db ${bundle_db} \
    --atomic-db ${atomic_db} \
    --freq-channel ${freq_channel} \
    --output-dir ${out_dir} \
    --sim-ids 0,39