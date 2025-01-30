import numpy as np
import healpy as hp
import argparse
import logging
import sqlite3
import os
import sys
import time
import sotodlib.mapmaking.demod_mapmaker as dmm
import sotodlib.preprocess.preprocess_util as pp_util

from sotodlib.mapmaking.noise_model import NmatUnit
from sotodlib.site_pipeline import preprocess_tod
from sotodlib.core.metadata import loader
from pixell import enmap
from mpi4py import MPI

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bundling'))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'misc'))
)
from coordinator import BundleCoordinator  # noqa
from mpi_utils import distribute_tasks  # noqa


def get_fullsky_geometry(res_arcmin=10., variant="fejer1"):
    """
    Generates a fullsky CAR template at resolution res-arcmin.
    """
    res = res_arcmin * np.pi/180/60
    return enmap.fullsky_geometry(res=res, proj='car', variant=variant)


def erik_make_map(obs, shape=None, wcs=None, nside=None, site=None):
    """
    Run the healpix mapmaker (currently in sotodlib sat-mapmaking-er-dev) on a
    given observation.

    Parameters
    ----------
        obs: axis manager
            Input TOD, after demodulation.
        shape: numpy.ndarray
            Shape of the output map geometry
        wcs: wcs
            WCS of the output map geometry
        nside: int
            Nside of the output map
        site: str
            Site of the observation, e.g. `so_sat1`
    Returns
    -------
        wmap: numpy.ndarray
            Output weighted TQU map, in healpix NESTED scheme
        weights: numpy.ndarray
            Output TQU weights, in healpix NESTED scheme
    """
    obs.wrap("weather", np.full(1, "toco"))
    obs.wrap("site", np.full(1, site))
    obs.flags.wrap(
        'glitch_flags',
        (obs.preprocess.turnaround_flags.turnarounds
         + obs.preprocess.jumps_2pi.jump_flag
         + obs.preprocess.glitches.glitch_flags)
    )
    mapmaker = dmm.setup_demod_map(NmatUnit(), shape=shape, wcs=wcs,
                                   nside=nside)
    mapmaker.add_obs('signal', obs)
    wmap = mapmaker.signals[0].rhs[0]
    weights = np.diagonal(mapmaker.signals[0].div[0], axis1=0, axis2=1)
    weights = np.moveaxis(weights, -1, 0)

    return wmap, weights


def make_map(obs, pix_type="hp", shape=None, wcs=None, nside=None, site=None,
             logger=None):
    """
    Run the healpix mapmaker (currently in sotodlib sat-mapmaking-er-dev) on a
    given observation.

    Parameters
    ----------
        obs: axis manager
            Input TOD, after demodulation.
        shape: numpy.ndarray
            Shape of the output map geometry (CAR only)
        wcs: wcs
            WCS of the output map geometry (CAR only)
        nside: int
            Nside of the output map (HEALPix only)
        site: str
            Site of the observation, e.g. `so_sat1`
    Returns
    -------
        wmap: numpy.ndarray
            Output weighted TQU map, in healpix NESTED scheme
        weights: numpy.ndarray
            Output TQU weights, in healpix NESTED scheme
    """
    obs.wrap("weather", np.full(1, "toco"))
    obs.wrap("site", np.full(1, site))
    if not hasattr(obs.flags, "glitch_flags"):
        obs.flags.wrap(
            'glitch_flags',
            (obs.preprocess.turnaround_flags.turnarounds
             + obs.preprocess.jumps_2pi.jump_flag
             + obs.preprocess.glitches.glitch_flags)
        )
    if pix_type == "car":
        nside = None
        assert (shape is not None) and (wcs is not None)
    elif pix_type == "hp":
        wcs, shape = (None, None)
        assert nside is not None
    mapmaker = dmm.setup_demod_map(
        NmatUnit(), shape=shape, wcs=wcs, nside=nside, split_labels=["science"]
    )
    mapmaker.add_obs('signal', obs)
    wmap = mapmaker.signals[0].rhs[0]
    weights = np.diagonal(mapmaker.signals[0].div[0], axis1=0, axis2=1)
    weights = np.moveaxis(weights, -1, 0)

    return wmap, weights


def get_logger(fmt=None, datefmt=None, debug=False, **kwargs):
    """Return logger from logging module
    code from pspipe

    Parameters
    ----------
        fmt: string
        the format string that preceeds any logging message
        datefmt: string
        the date format string
        debug: bool
        debug flag
    """
    # fmt = fmt or "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s" # noqa
    fmt = fmt or "%(asctime)s - %(message)s"
    datefmt = datefmt or "%d-%b-%y %H:%M:%S"
    logging.basicConfig(
        format=fmt,
        datefmt=datefmt,
        level=logging.DEBUG if debug else logging.INFO,
        force=True
    )
    return logging.getLogger(kwargs.get("name"))


def main(args):
    """
    """
    if args.pix_type not in ["hp", "car"]:
        raise ValueError(
            "Unknown pixel type, must be 'car' or 'hp'."
        )

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    logger = pp_util.init_logger("benchmark", verbosity=3)

    # ArgumentParser
    out_dir = args.output_directory
    logger.info(f"out_dir: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    plot_dir = f"{out_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    atomics_dir = f"{out_dir}/atomics_sims"
    os.makedirs(atomics_dir, exist_ok=True)

    # Databases
    bundle_db = args.bundle_db
    atom_db = args.atomic_db

    # Config files
    preprocess_config = args.preprocess_config
    preprocess_config_init = args.preprocess_config_init
    preprocess_config_proc = args.preprocess_config_proc
    do_multilayer_preproc = (preprocess_config_init is not None
                             and preprocess_config_proc is not None)

    logger.debug(f"atomic DB: {atom_db}")
    logger.debug(f"do_multilayer_preproc? {do_multilayer_preproc}")

    # Sim related arguments
    map_dir = args.map_dir
    map_string_format = args.map_string_format
    sim_ids = args.sim_ids

    # Pixelization arguments
    pix_type = args.pix_type
    if pix_type == "hp":
        nside = args.nside
        mfmt = ".fits"  # TODO: fits.gz for HEALPix
    elif pix_type == "car":
        if args.car_map_template is not None:
            w = enmap.read_map(args.car_map_template)
            shape = w.shape[-2:]
            wcs = w.wcs
        else:
            shape, wcs = get_fullsky_geometry()
        nside = None
        mfmt = ".fits"

    # Bundle query arguments
    freq_channel = args.freq_channel
    null_prop_val_inter_obs = args.null_prop_val_inter_obs
    bundle_id = args.bundle_id
    split_label = args.split_label_intra_obs

    # Extract list of ctimes from bundle database for the given
    # bundle_id - null split combination
    if os.path.isfile(bundle_db):
        logger.info(f"Loading from {bundle_db}.")
        bundle_coordinator = BundleCoordinator.from_dbfile(
            bundle_db, bundle_id=bundle_id,
            null_prop_val=null_prop_val_inter_obs
        )
    else:
        raise ValueError(f"DB file does not exist: {bundle_db}")

    ctimes = bundle_coordinator.get_ctimes(
        bundle_id=bundle_id,
        split_label=split_label,
        null_prop_val=null_prop_val_inter_obs,
    )

    # Read restrictive list of atomic-map metadata
    # (obs_id, wafer, freq_channel) from file, and intersect it
    # with the metadata in the bundling database.
    atomic_restrict = []
    if args.atomic_list is not None:
        atomic_restrict = list(
            map(tuple, np.load(args.atomic_list)["atomic_list"])
        )

    atomic_metadata = []
    db_con = sqlite3.connect(atom_db)
    db_cur = db_con.cursor()

    for ctime in ctimes:
        res = db_cur.execute(
            "SELECT obs_id, wafer FROM atomic WHERE "
            f"freq_channel == '{freq_channel}' AND ctime = '{ctime}' "
            f"AND split_label == '{split_label}'"
        )
        res = res.fetchall()

        for obs_id, wafer in res:
            atom_id = (obs_id, wafer, freq_channel)

            do_include = (args.atomic_list is None
                          or (atom_id in atomic_restrict))
            if do_include and not (atom_id in atomic_metadata):
                atomic_metadata.append((obs_id, wafer, freq_channel))
    db_con.close()

    logger.info(f"Using {len(atomic_metadata)} atomic maps to filter sims.")

    # Load preprocessing pipeline and extract from it list of preprocessing
    # metadata (detectors, samples, etc.) corresponding to each atomic map
    if do_multilayer_preproc:
        configs_init, ctx_init = pp_util.get_preprocess_context(
            preprocess_config_init
        )
        configs_proc, ctx_proc = pp_util.get_preprocess_context(
            preprocess_config_proc
        )
    else:
        configs, ctx = pp_util.get_preprocess_context(preprocess_config)

    # Distribute [natomics x nsims] tasks among [size] workers
    if "," in sim_ids:
        id_min, id_max = sim_ids.split(",")
    else:
        id_min = sim_ids
        id_max = id_min
    id_min = int(id_min)
    id_max = int(id_max)

    ids = np.arange(id_min, id_max+1)
    mpi_shared_list = [(i, j) for i in ids for j in atomic_metadata]
    task_ids = distribute_tasks(size, rank, len(mpi_shared_list),
                                logger=logger)
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    # Loop over local tasks (sim_id, atomic_id). For each of these, do:
    # * read simulated map
    # * load map into timestreams, apply preprocessing
    # * apply mapmaking
    local_wmaps = []
    local_weights = []
    local_labels = []

    for sim_id, (obs_id, wafer, freq) in local_mpi_list:
        start = time.time()
        map_fname = map_string_format.format(sim_id=sim_id)
        map_file = f"{map_dir}/{map_fname}"

        if args.pix_type == "car":
            logger.info(f"Loading CAR map: {map_file}")
            sim = enmap.read_map(map_file)
        elif args.pix_type == "hp":
            logger.info(f"Loading HP map: {map_file}")
            sim = hp.read_map(map_file, field=[0, 1, 2])
        else:
            raise ValueError("pix_type must be hp or car")

        logger.info(f"Loading {obs_id} {wafer} {freq} on sim {sim_id}")
        dets = {"wafer_slot": wafer, "wafer.bandpass": freq}

        ctx = ctx_proc if do_multilayer_preproc else ctx
        meta = ctx_proc.get_meta(obs_id=obs_id, dets=dets)

        # Focal plane thinning
        if args.fp_thin is not None:
            fp_thin = int(args.fp_thin)
            thinned = [m for im, m in enumerate(meta.dets.vals)
                       if im % fp_thin == 0]
            meta.restrict("dets", thinned)

        # Missing pointing not cut in preprocessing
        # meta.restrict(
        #     "dets", meta.dets.vals[~np.isnan(meta.focal_plane.gamma)]
        # )
        try:
            if do_multilayer_preproc:
                aman = pp_util.multilayer_load_and_preprocess_sim(
                    obs_id,
                    configs_init=configs_init,
                    configs_proc=configs_proc,
                    sim_map=sim,
                    meta=meta,
                    logger=logger
                )
            else:
                aman = preprocess_tod.load_preprocess_tod_sim(
                    obs_id,
                    sim_map=sim,
                    configs=configs,
                    meta=meta,
                    modulated=True,
                    logger=logger,
                    dets=dets,
                    # site="so_sat1",  # new field required from new from_map()
                    # ordering="RING"  # new field required for healpix
                )
        except loader.LoaderError:
            logger.info(
                f"ERROR: {obs_id} {wafer} {freq} metadata is not there. "
                "SKIPPING."
            )
            continue

        if aman.dets.count <= 1:
            continue

        # # OLD CODE
        # if pix_type == "car":
        #     filtered_sim = demod.make_map(
        #         aman,
        #         res=10*utils.arcmin,
        #         wcs_kernel=wcs,
        #     )
        #     wmap, w = filtered_sim["weighted_map"], filtered_sim["weight"]
        #     w = np.moveaxis(w.diagonal(), -1, 0)

        # elif pix_type == "hp":
        #     wmap, w = erik_make_map(aman, nside=nside, site="so_sat1")

        # NEW CODE
        wmap, w = make_map(aman, pix_type, shape=shape, wcs=wcs, nside=nside,
                           logger=logger)

        local_wmaps.append(wmap)
        local_weights.append(w)
        local_labels.append(sim_id)

        # Saving filtered atomics to disk
        logger.info(f"Rank {rank} saving labels {local_labels}")
        atomic_fname = map_string_format.format(sim_id=sim_id).replace(
            mfmt,
            f"_obsid{obs_id}_{wafer}_{freq_channel}_{split_label}{mfmt}"
        )
        f_wmap = f"{atomics_dir}/{atomic_fname.replace(mfmt, '_wmap' + mfmt)}"
        f_w = f"{atomics_dir}/{atomic_fname.replace(mfmt, '_w' + mfmt)}"

        if pix_type == "car":
            enmap.write_map(f_wmap, wmap)
            enmap.write_map(f_w, w)

        elif pix_type == "hp":
            hp.write_map(
                f_wmap, wmap, dtype=np.float32, overwrite=True, nest=True
            )
            hp.write_map(
                f_w, w, dtype=np.float32, overwrite=True, nest=True
            )
        end = time.time()
        logger.info(f"ELAPSED TIME for filtering: {end - start} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--atomic-db",
        help="Path to the atomic maps database."
    )
    parser.add_argument(
        "--atomic_list",
        help="Npz file with list of atomic maps to restrict the atomic_db to.",
        default=None
    )
    parser.add_argument(
        "--bundle-db",
        help="Path to the bundling database."
    )
    parser.add_argument(
        "--preprocess-config",
        help="Path to the preprocessing config file.",
        default=None
    )
    parser.add_argument(
        "--preprocess-config-init",
        help="Path to the init preprocessing config (multilayer) file.",
        default=None
    )
    parser.add_argument(
        "--preprocess-config-proc",
        help="Path to the proc preprocessing config (multilayer) file.",
        default=None
    )
    parser.add_argument(
        "--map-dir",
        help="Directory containing the maps to filter."
    )
    parser.add_argument(
        "--map_string_format",
        help="String formatting; must contain {sim_id}."
    )
    parser.add_argument(
        "--sim-ids",
        help="String of format 'sim_id_min,sim_id_max', or only 'sim_id'."
    )
    parser.add_argument(
        "--output-directory",
        help="Output directory for the filtered maps."
    )
    parser.add_argument(
        "--freq-channel",
        help="Frequency channel to filter."
    )
    parser.add_argument(
        "--bundle-id",
        type=int,
        default=0,
        help="Bundle ID to filter.",
    )
    parser.add_argument(
        "--null_prop_val_inter_obs",
        help="Null property value for inter-obs splits, e.g. 'pwv_low'.",
        default=None
    )
    parser.add_argument(
        "--split_label_intra_obs",
        help="Split label for intra-obs splits, e.g. 'scan_left'."
    )
    parser.add_argument(
        "--nside",
        help="Nside parameter for HEALPIX mapmaker.",
        type=int,
        default=512
    )
    parser.add_argument(
        "--pix_type",
        help="Pixelization type; 'hp' or 'car",
        default='hp'
    )
    parser.add_argument(
        "--car_map_template",
        help="path to CAR coadded (hits) map to be used as geometry template",
        default=None
    )
    parser.add_argument(
        "--fp-thin",
        help="Focal plane thinning factor",
        default=None
    )

    args = parser.parse_args()
    main(args)
