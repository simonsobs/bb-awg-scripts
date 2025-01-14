import numpy as np
import healpy as hp
import argparse
import logging
import sqlite3
import os
import yaml
import sys
import time
import sotodlib.mapmaking.demod_mapmaker as dmm

from sotodlib.mapmaking.noise_model import NmatUnit
from sotodlib.core import Context
from sotodlib.site_pipeline import preprocess_tod
from mpi4py import MPI

sys.path.append("/global/homes/k/kwolz/bbdev/bb-awg-scripts/pipeline/bundling")
sys.path.append("/global/homes/k/kwolz/bbdev/bb-awg-scripts/pipeline/misc")
from coordinator import BundleCoordinator  # noqa
from mpi_utils import distribute_tasks  # noqa


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
    #fmt = fmt or "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s" # noqa
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

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # ArgumentParser
    out_dir = args.output_directory
    os.makedirs(out_dir, exist_ok=True)

    plot_dir = f"{out_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    atomics_dir = f"{out_dir}/atomics_sims"
    os.makedirs(atomics_dir, exist_ok=True)

    # Databases
    atom_db = args.atomic_db
    bundle_db = args.bundle_db

    # Config files
    preprocess_config = args.preprocess_config

    # Sim related arguments
    nside = args.nside  # FIXME: generalize to CAR adn/or HEALPix
    map_dir = args.map_dir
    map_template = args.map_template
    sim_ids = args.sim_ids

    # Bundle query arguments
    freq_channel = args.freq_channel
    null_prop = args.null_prop
    bundle_id = args.bundle_id

    # Extract list of ctimes from bundle database for the given
    # bundle_id - null split combination
    bundle_db = BundleCoordinator.from_dbfile(
        bundle_db, bundle_id=bundle_id, null_prop_val=null_prop
    )
    ctimes = bundle_db.get_ctimes(bundle_id=bundle_id, null_prop_val=null_prop)

    # Extract list of atomic-map metadata (obs_id, wafer, freq_channel)
    # for the observations defined above
    atomic_metadata = []
    db_con = sqlite3.connect(atom_db)
    db_cur = db_con.cursor()
    for ctime in ctimes:
        res = db_cur.execute(
            "SELECT obs_id, wafer FROM atomic WHERE "
            f"freq_channel == '{freq_channel}' AND ctime == '{ctime}'"
        )
        res = res.fetchall()
        for obs_id, wafer in res:
            print("obs_id", obs_id, "wafer", wafer, freq_channel)
            atomic_metadata.append((obs_id, wafer, freq_channel))
    db_con.close()

    # Load preprocessing pipeline and extract from it list of preprocessing
    # metadata (detectors, samples, etc.) corresponding to each atomic map
    config = yaml.safe_load(open(preprocess_config, "r"))
    context = config["context_file"]
    ctx = Context(context)

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

    log = get_logger()
    task_ids = distribute_tasks(size, rank, len(mpi_shared_list), logger=log)
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
        map_fname = map_template.format(sim_id=sim_id)
        map_file = f"{map_dir}/{map_fname}"

        # FIXME: CAR version
        # sim = enmap.read_map(map_file)

        # HEALPix version
        sim = hp.read_map(map_file, field=[0, 1, 2])

        log.info(f"***** Doing {obs_id} {wafer} {freq} "
                 f"and SIMULATION {sim_id} *****")
        dets = {"wafer_slot": wafer, "wafer.bandpass": freq}
        meta = ctx.get_meta(obs_id, dets=dets)

        # Focal plane thinning
        if args.fp_thin is not None:
            fp_thin = int(args.fp_thin)
            thinned = [m for im, m in enumerate(meta.dets.vals)
                       if im % fp_thin == 0]
            meta.restrict("dets", thinned)

        # Missing pointing not cut in preprocessing
        meta.restrict(
            "dets", meta.dets.vals[~np.isnan(meta.focal_plane.gamma)]
        )

        aman = preprocess_tod.load_preprocess_tod_sim(
            obs_id,
            sim_map=sim,
            configs=config,
            meta=meta,
            modulated=True,
            site="so_sat1",  # new field required from new from_map function
            ordering="RING"  # new field required for healpix
        )
        log.info(f"Loaded {obs_id}, {wafer}, {freq}")

        if aman.dets.count <= 1:
            continue

        # FIXME: CAR version
        # filtered_sim = demod.make_map(
        #     aman,
        #     res=10*utils.arcmin,
        #     wcs_kernel=sim.wcs,
        # )
        # wmap, w = filtered_sim["weighted_map"], filtered_sim["weight"]
        # w = np.moveaxis(w.diagonal(), -1, 0)

        # HEALPix version
        wmap, w = erik_make_map(aman, nside=nside, site="so_sat1")

        local_wmaps.append(wmap)
        local_weights.append(w)
        local_labels.append(sim_id)

        # Saving filtered atomics to disk
        log.info(f"Rank {rank} saving labels {local_labels}")
        atomic_fname = map_template.format(sim_id=sim_id).replace(
            ".fits",
            f"_obsid{obs_id}_{wafer}_{freq_channel}.fits"
        )
        hp.write_map(
            f"{atomics_dir}/{atomic_fname.replace('.fits', '_wmap.fits')}",
            wmap, dtype=np.float32, overwrite=True, nest=True
        )
        hp.write_map(
            f"{atomics_dir}/{atomic_fname.replace('.fits', '_w.fits')}", w,
            dtype=np.float32, overwrite=True, nest=True
        )
        end = time.time()
        print(f"*** ELAPSED TIME for filtering: {end - start} seconds. ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--atomic-db",
        help="Path to the atomic maps database",
        type=str
    )
    parser.add_argument(
        "--bundle-db",
        help="Path to the bundle database",
        type=str
    )
    parser.add_argument(
        "--preprocess-config",
        help="Path to the preprocessing config file",
        type=str
    )
    parser.add_argument(
        "--map-dir",
        help="Directory containing the maps to filter",
        type=str
    )
    parser.add_argument(
        "--map-template",
        help="Template file for the map to filter",
        type=str
    )
    parser.add_argument(
        "--sim-ids",
        help="Comma separated list of simulation ids",
        type=str
    )
    parser.add_argument(
        "--output-directory",
        help="Output directory for the filtered maps",
        type=str
    )
    parser.add_argument(
        "--freq-channel",
        help="Frequency channel to filter",
        type=str
    )
    parser.add_argument(
        "--bundle-id",
        help="Bundle ID to filter",
    )
    parser.add_argument(
        "--null-prop",
        help="Null property to filter",
        default=None
    )
    parser.add_argument(
        "--nside",
        help="Nside parameter for HEALPIX mapmaker",
        type=int, default=512
    )
    parser.add_argument(
        "--fp-thin",
        help="Focal plane thinning factor",
        default=None
    )

    args = parser.parse_args()

    main(args)
