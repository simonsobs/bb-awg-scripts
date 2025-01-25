import sotodlib.preprocess.preprocess_util as pp_util
from pixell import enmap

obs_id = "obs_1722563197_satp3_1011111"
wafer = "ws2"
freq = "f150"
dets = {"wafer_slot": wafer, "wafer.bandpass": freq}

# These are the ones from pwg-scripts/sat-iso-review/preprocessing
preprocess_config_init = "/global/homes/k/kwolz/bbdev/bb-awg-scripts/users/kwolz/configs/preprocessing_config_20250108_sat-iso_init.yaml"  # noqa
preprocess_config_proc = "/global/homes/k/kwolz/bbdev/bb-awg-scripts/users/kwolz/configs/preprocessing_config_20250108_sat-iso_proc.yaml"  # noqa

configs_init, ctx_init = pp_util.get_preprocess_context(preprocess_config_init)
configs_proc, ctx_proc = pp_util.get_preprocess_context(preprocess_config_proc)

map_dir = "/global/cfs/cdirs/sobs/awg_bb/end_to_end_2412/tf_sims/pureT_fwhm30.0_sim0000_CAR.fits"  # noqa
sim = enmap.read_map(map_dir)

meta_proc = ctx_proc.get_meta(obs_id=obs_id, dets=dets)

# Focal plane thinning
fp_thin = 128
dets = [m for im, m in enumerate(meta_proc.dets.vals) if im % fp_thin == 0]

# Missing pointing not cut in preprocessing
# dets = meta_proc.dets.vals[~np.isnan(meta_proc.focal_plane.gamma)]
meta_proc.restrict("dets", dets)

logger = pp_util.init_logger("benchmark", verbosity=3)

# NOTE: If meta is not None, this fails!
pp_util.multilayer_load_and_preprocess_sim(
    obs_id, configs_init, configs_proc, sim_map=sim,
    meta=meta_proc, logger=logger
)
