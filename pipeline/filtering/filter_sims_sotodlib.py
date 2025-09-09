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
    for required_tag in ["{sim_id", "{sim_type"]:
        if required_tag not in args.sim_string_format:
            raise ValueError(f"sim_string_format does not have \
                             required placeholder {required_tag}")

    # MPI related initialization
    rank, size, comm = mpi.init(True)

    # Initialize the logger
    logger = pp_util.init_logger("benchmark", verbosity=3)

    # Path to databases
    bundle_dbs = {patch: args.bundle_db.format(patch=patch)
                  for patch in args.patches}
    atom_db = args.atomic_db

    # Pre-processing configuration files
    preprocess_config_init = args.preprocess_config_init
    preprocess_config_proc = args.preprocess_config_proc

    logger.debug(f"Using atomic DB from {atom_db}")

    # Sim related arguments
    atomic_sim_dir = args.atomic_sim_dir
    sim_dir = args.sim_dir
    sim_string_format = args.sim_string_format
    sim_ids = args.sim_ids
    pure_types = [f"pure{i}" for i in "TEB"] if "{pure_type}" in sim_string_format else ["signal map"]  # noqa

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
    for freq_channel, patch in product(args.freq_channels, args.patches):
        atomics_dir[(patch, freq_channel)] = {}
        out_dir = args.output_dir.format(patch=patch,
                                         freq_channel=freq_channel)
        for sim_id in sim_ids:
            atomics_dir[patch, freq_channel][sim_id] = f"{out_dir}/atomic_sims/{sim_id:04d}"  # noqa
            os.makedirs(atomics_dir[patch, freq_channel][sim_id],
                        exist_ok=True)

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
    ctimes = {patch: None for patch in args.patches}
    for patch in args.patches:
        if os.path.isfile(bundle_dbs[patch]):
            logger.info(f"Loading from {bundle_dbs[patch]}.")
            bundle_coordinator = coord.BundleCoordinator.from_dbfile(
                bundle_dbs[patch], bundle_id=bundle_id
            )
        else:
            raise ValueError(f"DB file does not exist: {bundle_dbs[patch]}")

        # Extract all ctimes for the given bundle_id
        ctimes[patch] = bundle_coordinator.get_ctimes(bundle_id=bundle_id)

    # Connect the the atomic map DB
    atomic_metadata = []
    db_con = sqlite3.connect(atom_db)
    db_cur = db_con.cursor()

    # TODO: check if query_restrict is channel- or patch-specific
    query_restrict = args.query_restrict
    queries = {
        (patch, freq_channel): fu.get_query_atomics(
            freq_channel, ctimes[patch], query_restrict=query_restrict
        )
        for patch, freq_channel in product(args.patches, args.freq_channels)
    }

    atomic_metadata = {split_label: [] for split_label in intra_obs_splits}
    atomic_metadata["science"] = []
    for (patch, freq_channel), query in queries.items():
        res_science = db_cur.execute(query)
        res_science = res_science.fetchall()
        for obs_id, wafer in res_science:
            atomic_metadata["science"] += [(patch, freq_channel,
                                            obs_id, wafer)]
        for split_label in intra_obs_splits:
            query = fu.get_query_atomics(freq_channel, ctimes[patch],
                                         split_label=split_label,
                                         query_restrict=query_restrict)
            res_split = db_cur.execute(query)
            res_split = res_split.fetchall()
            for obs_id, wafer in res_split:
                if (obs_id, wafer) in res_science:
                    atomic_metadata[split_label] += [
                        (patch, freq_channel, obs_id, wafer)
                    ]
        logger.info(
            f"{patch}, {freq_channel}, 'science': "
            f"{len(res_science)} atomic maps to filter."
        )
    db_con.close()

    # Load preprocessing pipeline and extract from it list of preprocessing
    # metadata (detectors, samples, etc.) corresponding to each atomic map
    configs_init, _ = pp_util.get_preprocess_context(
        preprocess_config_init
    )
    configs_proc, ctx_proc = pp_util.get_preprocess_context(
        preprocess_config_proc
    )

    # Initialize tasks for MPI sharing
    mpi_shared_list = atomic_metadata["science"]

    # Every rank must have the same shared list
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list),
                                    logger=logger)
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    # Loop over set of local tasks (patch, freq_channel, obs_id, wafer).
    # For each of these, loop over (sim_id, sim_type) and do:
    # * read simulated map
    # * load map into timestreams, apply preprocessing
    # * apply mapmaking
    for patch, freq_channel, obs_id, wafer in local_mpi_list:
        start = time.time()

        # Get axis manager metadata for the given obs
        dets = {"wafer_slot": wafer, "wafer.bandpass": freq_channel}

        try:
            meta = ctx_proc.get_meta(obs_id=obs_id, dets=dets)
        except loader.LoaderError:
            logger.warning(f"NO METADATA: "
                           f"({patch}, {freq_channel}, {obs_id}, {wafer})")
            continue

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
        data_aman = pp_util.multilayer_load_and_preprocess(
            obs_id,
            configs_init,
            configs_proc,
            meta=meta,
            logger=logger,
            init_only=True,
        )

        for sim_id, sim_type in product(sim_ids, args.sim_types):

            logger.info(f"Loading ({patch}, {freq_channel}, {obs_id}, {wafer})"
                        f" to filter {sim_type}, sim {sim_id}")
            start0 = time.time()

            # Path to simulation
            map_fname = sim_string_format.format(sim_id=sim_id,
                                                 sim_type=sim_type)
            map_file = f"{sim_dir}/{map_fname}"

            # Handling pixellization
            if args.pix_type == "car":
                logger.debug(f"Loading CAR map: {map_file}")
                sim = enmap.read_map(map_file)
            elif args.pix_type == "hp":
                logger.debug(f"Loading HP map: {map_file}")
                sim = hp.read_map(map_file, field=[0, 1, 2])
            else:
                raise ValueError("pix_type must be hp or car")

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
                logger.warning(
                    "METADATA MISSING: "
                    f"({patch}, {freq_channel}, {obs_id}, {wafer}) "
                    "SKIPPING."
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
                if (patch, freq_channel, obs_id, wafer) in atomic_metadata[split_label]:  # noqa
                    wmap = wmap_dict[split_label]
                    w = weights_dict[split_label]

                    # Saving filtered atomics to disk
                    atomic_fname = map_fname.replace(
                        mfmt,
                        f"_{obs_id}_{wafer}_{split_label}{mfmt}"
                    )

                    f_wmap = atomics_dir[patch, freq_channel][sim_id]
                    f_wmap += f"/{atomic_fname.replace(mfmt, '_wmap' + mfmt)}"
                    f_w = f_wmap.replace(mfmt, '_weights' + mfmt)

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
            end0 = time.time()
            logger.info(f"Filtered in {end0 - start0:.1f} seconds: "
                        f"{sim_type}, sim {sim_id} with setup "
                        f"({patch}, {freq_channel}, {obs_id}, {wafer})")

        logger.info(f"Processed 3 x {len(sim_ids)} simulations for "
                    f"({patch}, {freq_channel}, {obs_id}, {wafer}) in "
                    f"{time.time() - start:.1f} seconds.")
    comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, help="yaml file with configuration."
    )

    args = parser.parse_args()
    config = fu.Cfg.from_yaml(args.config_file)
    main(config)
