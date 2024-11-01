atomic_maps_list=atomic_map_list.txt
preprocess_config=/global/cfs/cdirs/sobs/users/rosenber/preprocess/satp1_241029/minimal_preprocess.yaml
map_dir=/pscratch/sd/a/alaposta/bb_awg_241030/bb-awg-scripts/users/alaposta/tf_sims
sim_ids=0,3
output_directory="test_filter_hp_241030"
map_templates=("pureE_nside256_fwhm30.0_sim{sim_id:04d}.fits")

for map_template in "${map_templates[@]}";
do
    time srun -n 4 -c 64 --cpu_bind=cores python filter_sotodlib_healpix.py \
        --atomic-maps-list $atomic_maps_list \
        --preprocess-config $preprocess_config \
        --map-dir $map_dir \
        --map-template $map_template \
        --sim-ids $sim_ids \
        --output-directory $output_directory
done