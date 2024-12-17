nside=512
n_sims=50
out_dir=/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/cmb_sims/filtered_science
bundles=(0 1)

for bundle in "${bundles[@]}";
do
    echo "bundle ${bundle}"
    map_fname_format="cmb_nside512_fwhm30.0_sim{sim_id:04}_bundle${bundle}_filtered.fits"
    alm_fname_format="alm_cmb_lmax1535_fwhm30.0_{sim_id:04}_bundle${bundle}_filtered.fits"
    srun -n 50 -c 10 --cpu_bind=cores python map2alm_sims.py \
        --nside $nside \
        --n_sims $n_sims \
        --out_dir $out_dir \
        --map_fname_format $map_fname_format \
        --alm_fname_format $alm_fname_format
done
