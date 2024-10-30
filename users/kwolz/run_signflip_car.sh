# queries_list="freq_channel = 'f090'" "elevation >= '50'" "ctime = '1710705196'"
outdir=/pscratch/sd/k/kwolz/bbdev/bb-awg-scripts/bundle_signflip/test_20241030
nbundles=4

python ../../pipeline/bundling/get_atomics_list.py \
    --map_dir /pscratch/sd/s/susannaz/shared/SO_ISO/satp3_maps/cmb_map_satp3_8May2024 \
    --queries_list "freq_channel = 'f090'" "elevation >= '50'" \
    --outdir $outdir

srun -n $nbundles python ../../pipeline/bundling/generate_map_bundles.py \
    --atomic_maps_list ${outdir}/atomic_maps_list.npz \
    --pixelization car \
    --seed 1234 \
    --nbundles $nbundles \
    --do_signflip \
    --atomic_maps_weights ${outdir}/bundle{id_bundle}_atomics.npz \
    --outdir $outdir