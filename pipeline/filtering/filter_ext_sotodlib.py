"""
Based on filter_sims_sotodlib.py, 
adapted to run on the external data.
"""
import numpy as np
import healpy as hp
import argparse
import sqlite3
import os
import sys
import time
from itertools import product

import sotodlib.preprocess.preprocess_util as pp_util
from sotodlib.core.metadata import loader
from pixell import enmap

# TODO: Make it an actual module
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bundling'))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'misc'))
)
import coordinator as coord  # noqa
import filtering_utils as fu  # noqa
import mpi_utils as mpi  # noqa


def main(args):
    """
    """
    if args.pix_type not in ["hp", "car"]:
        raise ValueError(
            "Unknown pixel type, must be 'car' or 'hp'."
        )
    # for required_tag in ["{sim_id", "{pure_type}"]:
    #     if required_tag not in args.sim_string_format:
    #         raise ValueError(f"sim_string_format does not have \
    #                          required placeholder {required_tag}")

    # MPI related initialization
    rank, size, comm = mpi.init(True)

    # Initialize the logger
    logger = pp_util.init_logger("benchmark", verbosity=3)

    # Where to store outputs
    out_dir = args.output_dir
    logger.info(f"Outputs to be written to {out_dir}")

    # Path to databases
    bundle_db = args.bundle_db
    atom_db = args.atomic_db

    # Pre-processing configuration files
    preprocess_config_init = args.preprocess_config_init
    preprocess_config_proc = args.preprocess_config_proc

    logger.debug(f"Using atomic DB from {atom_db}")

    # Sim related arguments
    sim_dir = args.sim_dir
    sim_string_format = args.sim_string_format
    sim_ids = args.sim_ids

    # Creating the simulation indices range to filter
    if isinstance(sim_ids, list):
        sim_ids = np.array(sim_ids, dtype=int)
    elif isinstance(sim_ids, str):
        if "," in sim_ids:
            id_min, id_max = sim_ids.split(",")
            sim_ids = np.arange(int(id_min), int(id_max)+1)
        else:
            sim_ids = np.array([int(sim_ids)])
    else:
        raise ValueError("Argument 'sim_ids' has the wrong format")

    # Create output directories
    atomics_dir = {}
    for sim_id in sim_ids:
        atomics_dir[sim_id] = f"{out_dir}/atomic_maps"  # noqa
        os.makedirs(atomics_dir[sim_id], exist_ok=True)

    # Arguments related to pixellization
    pix_type = args.pix_type
    if pix_type == "hp":
        nside = args.nside
        mfmt = ".fits"  # TODO: fits.gz for HEALPix
    elif pix_type == "car":
        if args.car_map_template is not None:
            _, wcs = enmap.read_map_geometry(args.car_map_template)
        else:
            _, wcs = fu.get_fullsky_geometry() # Could be problematic if using all default values! # noqa
        nside = None
        mfmt = ".fits"

    # Bundle query arguments
    freq_channel = args.freq_channel
    bundle_id = args.bundle_id

    # Gather all intra obs split labels in a list
    intra_obs_splits = args.intra_obs_splits
    if isinstance(intra_obs_splits, str):
        if "," not in intra_obs_splits:
            intra_obs_splits = [intra_obs_splits]
        else:
            intra_obs_splits = intra_obs_splits.split(",")

    # Extract list of ctimes from bundle database for the given
    # bundle_id - null split combination
    if os.path.isfile(bundle_db):
        logger.info(f"Loading from {bundle_db}.")
        bundle_coordinator = coord.BundleCoordinator.from_dbfile(
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

    query_restrict = args.query_restrict
    query = fu.get_query_atomics(freq_channel, ctimes,
                                 query_restrict=query_restrict)

    res = db_cur.execute(query)
    res = res.fetchall()
    atomic_metadata = {
        "science": [
            (obs_id, wafer) for obs_id, wafer in res
        ]
    }
    for split_label in intra_obs_splits:
        query = fu.get_query_atomics(freq_channel, ctimes,
                                     split_label=split_label)
        res = db_cur.execute(query)
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

    # Every rank must have the same shared list
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list),
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

        for sim_id, pure_type in product(sim_ids, [f"pure{i}" for i in "TEB"]):
            logger.info(f"Processing {pure_type} for sim_id {sim_id:04d}")
            # Initialize a timer
            start0 = time.time()
            # Path to simulation
            map_fname = sim_string_format.format(sim_id=sim_id,
                                                 pure_type=pure_type)
            map_file = f"{sim_dir}/{map_fname}"
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
                        f"on {pure_type}, sim {sim_id}")

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
            wmap_dict, weights_dict = fu.make_map_wrapper(
                aman, intra_obs_splits, pix_type, shape=None, wcs=wcs,
                nside=nside, logger=logger
            )

            for split_label in intra_obs_splits:
                # We save only files for which we actually have data for a
                # given null split
                if (obs_id, wafer) in atomic_metadata[split_label]:
                    wmap = wmap_dict[split_label]
                    w = weights_dict[split_label]

                    # Saving filtered atomics to disk
                    atomic_fname = map_fname.replace(
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
                f"Filtered {pure_type} sim {sim_id:04d} in "
                f"{end0 - start0:.1f} seconds."
            )
        logger.info(f"Processed 3 x {len(sim_ids)} simulations for "
                    f"{obs_id} {wafer} in {time.time() - start:.1f} seconds.")
    comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, help="yaml file with configuration."
    )

    args = parser.parse_args()
    config = fu.Cfg.from_yaml(args.config_file)
    main(config)
