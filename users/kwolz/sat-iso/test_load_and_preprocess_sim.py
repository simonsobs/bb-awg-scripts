import sotodlib.preprocess.preprocess_util as pp_util
from pixell import enmap
from sotodlib.site_pipeline import preprocess_tod

obs_id = "obs_1711030907_satp3_1111111"
wafer = "ws0"
freq = "f150"
dets = {"wafer_slot": wafer, "wafer.bandpass": freq}

# These are the ones from pwg-scripts/sat-iso-review/preprocessing
preprocess_config = "/global/homes/k/kwolz/bbdev/bb-awg-scripts/users/kwolz/configs/minimal_preprocess_susanna_satp3.yaml"
configs, ctx = pp_util.get_preprocess_context(preprocess_config, None)

map_dir = "/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/tf_sims/pureT_fwhm30.0_sim0000_CAR.fits"
sim = enmap.read_map(map_dir)

meta = ctx.get_meta(obs_id=obs_id, dets=dets)

# Focal plane thinning
fp_thin = 128
dets = [m for im, m in enumerate(meta.dets.vals) if im % fp_thin == 0]

# # Missing pointing not cut in preprocessing
# dets = meta_proc.dets.vals[~np.isnan(meta_proc.focal_plane.gamma)]
# meta_proc.restrict(
#     "dets", dets
# )

logger = pp_util.init_logger("benchmark", verbosity=3)

aman = preprocess_tod.load_preprocess_tod_sim(
    obs_id,
    sim_map=sim,
    configs=configs,
    meta=meta,
    modulated=True,
    logger=logger,
    dets=dets
    # site="so_sat1",  # new field required from new from_map()
    # ordering="RING"  # new field required for healpix
)
