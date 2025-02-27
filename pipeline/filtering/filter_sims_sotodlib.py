import numpy as np
import healpy as hp
import argparse
import sqlite3
import os
import sys
import time
import sotodlib.preprocess.preprocess_util as pp_util

from sotodlib.core.metadata import loader
from pixell import enmap
from sotodlib.coords import P
from mpi4py import MPI

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bundling'))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'misc'))
)
from coordinator import BundleCoordinator  # noqa
from mpi_utils import distribute_tasks  # noqa
from sotodlib.coords.demod import make_map  # noqa


def get_fullsky_geometry(res_arcmin=5., variant="fejer1"):
    """
    Generates a fullsky CAR template at resolution res-arcmin.
    """
    res = res_arcmin * np.pi/180/60
    return enmap.fullsky_geometry(res=res, proj='car', variant=variant)


def make_map_wrapper(obs, split_labels, pix_type="hp", shape=None, wcs=None,
                     nside=None, site=None, logger=None):
    """
    """
    obs.wrap("weather", np.full(1, "toco"))
    obs.wrap("site", np.full(1, site))
    if pix_type == "car":
        nside = None
        assert wcs is not None
    elif pix_type == "hp":
        wcs = None
        assert nside is not None

    inv_var = 1 / obs.preprocess.noiseQ_nofit.white_noise ** 2

    wmap_dict = {}
    weights_dict = {}

    for n_split, split_label in enumerate(split_labels):
        cuts = obs.flags.glitch_flags + ~obs.preprocess.split_flags.cuts[split_label]  # noqa

        Proj = P.for_tod(obs, wcs_kernel=wcs, comps='TQU', cuts=cuts,
                         hwp=True, interpol=None)
        result = make_map(obs, P=Proj, det_weights=2 * inv_var,
                          det_weights_demod=inv_var)
        wmap_dict[split_label] = result['weighted_map']
        weights_dict[split_label] = result['weight']
        # transform (3, 3, N, n) array to (3, N, n) keeping only diagonals
        # in the first two dimensions
        weights_dict[split_label] = np.moveaxis(
            weights_dict[split_label].diagonal(), -1, 0
        )

    return wmap_dict, weights_dict


def main(args):
    """
    """
    if args.pix_type not in ["hp", "car"]:
        raise ValueError(
            "Unknown pixel type, must be 'car' or 'hp'."
        )

    # MPI related initialization
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    success = True

    # Initialize the logger
    logger = pp_util.init_logger("benchmark", verbosity=3)

    # Where to store outputs
    out_dir = args.output_directory
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Outputs to be written to {out_dir}")

    # Path to databases
    bundle_db = args.bundle_db
    atom_db = args.atomic_db

    # Pre-processing configuration files
    preprocess_config_init = args.preprocess_config_init
    preprocess_config_proc = args.preprocess_config_proc

    logger.debug(f"Using atomic DB from {atom_db}")

    # Sim related arguments
    map_dir = args.map_dir
    map_string_format = args.map_string_format
    sim_ids = args.sim_ids

    # Creating the simulation indices range to filter
    if "," in sim_ids:
        id_min, id_max = sim_ids.split(",")
        sim_ids = np.arange(int(id_min), int(id_max)+1)
    else:
        sim_ids = np.array([int(sim_ids)])

    # Create output directories
    atomics_dir = {}
    for sim_id in sim_ids:
        atomics_dir[sim_id] = f"{out_dir}/{sim_id:04d}/{args.freq_channel}"
        os.makedirs(atomics_dir[sim_id], exist_ok=True)

    # Arguments related to pixellization
    pix_type = args.pix_type
    if pix_type == "hp":
        nside = args.nside
        mfmt = ".fits"  # TODO: fits.gz for HEALPix
    elif pix_type == "car":
        if args.car_map_template is not None:
            shape, wcs = enmap.read_map_geometry(args.car_map_template)
        else:
            shape, wcs = get_fullsky_geometry() # Could be problematic if using all default values! # noqa
        nside = None
        mfmt = ".fits"

    # Bundle query arguments
    freq_channel = args.freq_channel
    bundle_id = args.bundle_id
    # Gather all intra obs split labels in a list
    if "," not in args.split_labels_intra_obs:
        split_labels = [args.split_labels_intra_obs]
    else:
        split_labels = args.split_labels_intra_obs.split(",")

    # Extract list of ctimes from bundle database for the given
    # bundle_id - null split combination
    if os.path.isfile(bundle_db):
        logger.info(f"Loading from {bundle_db}.")
        bundle_coordinator = BundleCoordinator.from_dbfile(
            bundle_db, bundle_id=bundle_id
        )
    else:
        raise ValueError(f"DB file does not exist: {bundle_db}")

    # Extract all ctimes for the given bundle_id
    ctimes = bundle_coordinator.get_ctimes(bundle_id=bundle_id)

    # Connect the the atomic map DB
    atomic_metadata = []
    db_con = sqlite3.connect(atom_db)
    db_cur = db_con.cursor()

    query = f"""
            SELECT obs_id, wafer
            FROM atomic
            WHERE freq_channel == '{freq_channel}'
            AND ctime IN {tuple(ctimes)}
            AND split_label == 'science'
            """
    query_restrict = args.query_restrict
    if query_restrict:
        query += f" AND {query_restrict}"
    res = db_cur.execute(query)
    res = res.fetchall()
    atomic_metadata = {
        "science": [
            (obs_id, wafer) for obs_id, wafer in res
        ]
    }
    for split_label in split_labels:
        res = db_cur.execute(
            f"""
            SELECT obs_id, wafer
            FROM atomic
            WHERE freq_channel == '{freq_channel}'
            AND ctime IN {tuple(ctimes)}
            AND split_label == '{split_label}'
            AND valid == 1
            """
        )
        res = res.fetchall()
        atomic_metadata[split_label] = [
            (obs_id, wafer) for obs_id, wafer in res
            if (obs_id, wafer) in atomic_metadata["science"]
        ]
    db_con.close()

    logger.info(f"{len(atomic_metadata['science'])} atomic maps to filter.")

    # Load preprocessing pipeline and extract from it list of preprocessing
    # metadata (detectors, samples, etc.) corresponding to each atomic map
    configs_init, _ = pp_util.get_preprocess_context(
        preprocess_config_init
    )
    configs_proc, ctx_proc = pp_util.get_preprocess_context(
        preprocess_config_proc
    )

    # Initialize tasks for MPI sharing
    # Removing sim_id from the MPI loop
    mpi_shared_list = atomic_metadata["science"]
    task_ids = distribute_tasks(size, rank, len(mpi_shared_list),
                                logger=logger)
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    # Loop over local tasks (sim_id, atomic_id). For each of these, do:
    # * read simulated map
    # * load map into timestreams, apply preprocessing
    # * apply mapmaking
    for obs_id, wafer in local_mpi_list:
        # Get axis manager metadata for the given obs
        dets = {"wafer_slot": wafer, "wafer.bandpass": freq_channel}
        meta = ctx_proc.get_meta(obs_id=obs_id, dets=dets)

        # Focal plane thinning
        if args.fp_thin is not None:
            fp_thin = int(args.fp_thin)
            thinned = [
                m for im, m in enumerate(meta.dets.vals)
                if im % fp_thin == 0
            ]
            meta.restrict("dets", thinned)

        # Process data here to have t2p leakage template
        # Only need to run it once for all simulations
        # and only the pre-demodulation part.
        print(obs_id, wafer)
        data_aman = pp_util.multilayer_load_and_preprocess(
            obs_id,
            configs_init,
            configs_proc,
            meta=meta,
            logger=logger,
            init_only=True
        )

        start = time.time()
        logger.info(f"Processing {obs_id} {wafer}")

        for sim_id in sim_ids:
            logger.info(f"Processing sim_id {sim_id:04d}")
            # Initialize a timer
            start0 = time.time()
            # Path to simulation
            map_fname = map_string_format.format(sim_id=sim_id)
            map_file = f"{map_dir}/{map_fname}"
            # Handling pixellization
            if args.pix_type == "car":
                logger.info(f"Loading CAR map: {map_file}")
                sim = enmap.read_map(map_file)
            elif args.pix_type == "hp":
                logger.info(f"Loading HP map: {map_file}")
                sim = hp.read_map(map_file, field=[0, 1, 2])
            else:
                raise ValueError("pix_type must be hp or car")

            logger.info(f"Loading {obs_id} {wafer} {freq_channel} "
                        f"on sim {sim_id}")

            try:
                aman = pp_util.multilayer_load_and_preprocess_sim(
                    obs_id,
                    configs_init=configs_init,
                    configs_proc=configs_proc,
                    sim_map=sim,
                    meta=meta,
                    logger=logger,
                    t2ptemplate_aman=data_aman
                )

            except loader.LoaderError:
                logger.info(
                    f"ERROR: {obs_id} {wafer} {freq_channel} metadata is not "
                    "there. SKIPPING."
                )
                continue

            if aman.dets.count <= 1:
                continue

            # Run the mapmaker
            wmap_dict, weights_dict = make_map_wrapper(
                aman, split_labels, pix_type, shape=None, wcs=wcs, nside=nside,
                logger=logger
            )

            for split_label in split_labels:
                # We save only files for which we actually have data for a
                # given null split
                if (obs_id, wafer) in atomic_metadata[split_label]:
                    wmap = wmap_dict[split_label]
                    w = weights_dict[split_label]

                    # Saving filtered atomics to disk
                    atomic_fname = map_string_format.format(sim_id=sim_id).replace(  # noqa
                        mfmt,
                        f"_{obs_id}_{wafer}_{freq_channel}_{split_label}{mfmt}"
                    )

                    f_wmap = f"{atomics_dir[sim_id]}/{atomic_fname.replace(mfmt, '_wmap' + mfmt)}"  # noqa
                    f_w = f"{atomics_dir[sim_id]}/{atomic_fname.replace(mfmt, '_weights' + mfmt)}"  # noqa

                    if pix_type == "car":
                        enmap.write_map(f_wmap, wmap)
                        enmap.write_map(f_w, w)

                    elif pix_type == "hp":
                        hp.write_map(
                            f_wmap, wmap, dtype=np.float32, overwrite=True,
                            nest=True
                        )
                        hp.write_map(
                            f_w, w, dtype=np.float32, overwrite=True, nest=True
                        )
            # Stop timer
            end0 = time.time()
            logger.info(
                f"Filtering sim {sim_id:04d} in {end0 - start0:.1f} seconds."
            )
        logger.info(f"Processed {len(sim_ids)} simulations for "
                    f"{obs_id} {wafer} in {time.time() - start:.1f} seconds.")

    success = all(comm.allgather(success))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--atomic-db",
        help="Path to the atomic maps database."
    )
    parser.add_argument(
        "--bundle-db",
        help="Path to the bundling database."
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
        "--query_restrict",
        help="SQL query to restrict obs from the atomic database.",
        default=""
    )
    parser.add_argument(
        "--split_labels_intra_obs",
        help="Comma-separated list of split label for intra-obs splits, "
             "e.g. 'scan_left,scan_right,det_left,det_right'.",
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
