# queries_list="freq_channel = 'f090'" "elevation >= '50'" "ctime = '1710705196'" "wafer_slot = 'ws0'"
outdir=/global/cfs/cdirs/sobs/users/rosenber/maps/satp1_241031/coadd
nbundles=4
map_dir=/global/cfs/cdirs/sobs/users/rosenber/maps/satp1_241031/atomics

python ../../pipeline/bundling/get_atomics_list.py \
    --map_dir ${map_dir} \
    --queries_list "freq_channel = 'f150'" \
    --outdir $outdir \
    --db_fpath ${map_dir}/atomic_maps_satp1_241031.db \
    --ext "fits.gz" \
    --verbose

srun -n $nbundles python ../../pipeline/bundling/generate_map_bundles.py \
    --atomic_maps_list ${outdir}/atomic_maps_list.npz \
    --pixelization hp \
    --seed 1234 \
    --nbundles $nbundles \
    --outdir $outdir
