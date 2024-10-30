# queries_list="freq_channel = 'f090'" "elevation >= '50'" "ctime = '1710705196'"
outdir=/pscratch/sd/k/kwolz/bbdev/bb-awg-scripts/bundle_signflip/test_20241029
nbundles=4

python ../../pipeline/bundling/get_atomics_list.py \
    --map_dir /pscratch/sd/r/rosenber/so_pwg/hp_mapmaker/satp3_run12_hp_240705 \
    --queries_list "wafer_slot = 'ws0'" "freq_channel = 'f090'" \
    --outdir $outdir

srun -n $nbundles python ../../pipeline/bundling/generate_map_bundles.py \
    --atomic_maps_list ${outdir}/atomic_maps_list.npz \
    --pixelization hp \
    --seed 1234 \
    --nbundles $nbundles \
    --outdir $outdir \
    --do_signflip \
    --atomic_maps_weights ${outdir}/bundle{id_bundle}_atomics.npz \
    --outdir $outdir
