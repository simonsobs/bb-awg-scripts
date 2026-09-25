import numpy as np
import argparse
import sqlite3
import os
import sys
import time
from itertools import product

import sotodlib.preprocess.preprocess_util as pp_util
from sotodlib.core.metadata import loader

# TODO: Make it an actual module
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bundling'))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'misc'))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

import coordinator as coord  # noqa
import filtering_utils as fu  # noqa
import mpi_utils as mpi  # noqa
from bundling_utils import read_map, write_map
from configs import Cfg

def main(args):
    """
    """
    # MPI related initialization
    rank, size, comm = mpi.init(True)

    # Initialize the logger
    logger = pp_util.init_logger("benchmark", verbosity=args.verbosity)
    if rank == 0:
        start = time.time()

    sim_types, sim_ids, sim_dir, sim_string_format = fu.process_sim_args(args, rank, logger)
    freq_channels = np.atleast_1d(args.freq_channel)
    patches = np.atleast_1d(args.patch)
    atom_db = args.atomic_db

    # Pre-processing configuration files
    preprocess_config_init = args.filtering.preprocess_config_init
    preprocess_config_proc = args.filtering.preprocess_config_proc

    logger.debug(f"Using atomic DB from {atom_db}")

    # Create output directories
    atomics_dir = {}
    for freq_channel, patch in product(freq_channels, patches):
        atomics_dir[(patch, freq_channel)] = {}
        for sim_type in sim_types:
            for sim_id in sim_ids:
                atomic_sim_dir = args.filtering.atomic_sim_dir.format(
                    patch=patch, freq_channel=freq_channel,
                    sim_type=sim_type, sim_id=sim_id
                )
                dir_key = sim_id if sim_id is not None else sim_type
                atomics_dir[patch, freq_channel][dir_key] = atomic_sim_dir  # noqa
                if sim_id is not None and "{sim_id" not in args.filtering.atomic_sim_dir:
                    atomics_dir[patch, freq_channel][dir_key] += f"/{sim_id:04d}"  # noqa
                os.makedirs(atomics_dir[patch, freq_channel][dir_key],
                            exist_ok=True)

    # Arguments related to pixellization
    pix_type, mfmt, car_map_template, nside, wcs = fu.get_pix_type_args(args)

    # Bundle query arguments
    bundle_id = args.filtering.bundle_id

    # Gather all intra obs split labels in a list
    intra_obs_splits = args.intra_obs_splits
    if isinstance(intra_obs_splits, str):
        if "," not in intra_obs_splits:
            intra_obs_splits = [intra_obs_splits]
        else:
            intra_obs_splits = intra_obs_splits.split(",")
    if not len(intra_obs_splits):
        # If no intra-obs splits are given to coadd, use the default ones.
        intra_obs_splits = ["scan_left", "scan_right"]

    # Extract list of ctimes from bundle database for the given
    # bundle_id - null split combination
    ctimes = {patch: None for patch in patches}
    queries = {}
    atomic_metadata = {split_label: [] for split_label in intra_obs_splits}
    atomic_metadata["science"] = []

    for patch in patches:
        args_patch = args.child(patch=patch)
        bundle_db = args_patch.bundle_db_full
        if os.path.isfile(bundle_db):
            logger.info(f"Loading from {bundle_db}.")
            bundle_coordinator = coord.BundleCoordinator.from_dbfile(
                bundle_db, bundle_id=bundle_id)
        else:
            raise ValueError(f"DB file does not exist: {bundle_db}")

        # Extract all ctimes for the given bundle_id
        ctime = bundle_coordinator.get_ctimes(bundle_id=bundle_id)
        ctimes[patch] = ctime

        query_restrict = args_patch.query_restrict_patch
        for freq_channel in freq_channels:
            query = fu.get_query_atomics(freq_channel, ctime, query_restrict=query_restrict)
            queries[(patch, freq_channel)] = query

            db_cur = sqlite3.connect(atom_db).cursor()
            res_science = db_cur.execute(query)
            res_science = res_science.fetchall()
            db_cur.close()

            for obs_id, wafer in res_science:
                atomic_metadata["science"] += [(patch, freq_channel,
                                                obs_id, wafer)]
            for split_label in intra_obs_splits:
                query = fu.get_query_atomics(freq_channel, ctime,
                                             split_label=split_label,
                                             query_restrict=query_restrict)
                db_cur = sqlite3.connect(atom_db).cursor()
                res_split = db_cur.execute(query)
                res_split = res_split.fetchall()
                db_cur.close()
                for obs_id, wafer in res_split:
                    if (obs_id, wafer) in res_science:
                        atomic_metadata[split_label] += [
                            (patch, freq_channel, obs_id, wafer)
                        ]
            logger.info(
                f"{patch}, {freq_channel}, 'science': "
                f"{len(res_science)} atomic maps to filter."
            )

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

    # Ensure that idle workers finish and don't hang
    if not task_ids:
        comm.barrier()

    # Loop over set of local tasks (patch, freq_channel, obs_id, wafer).
    # For each of these, loop over (sim_id, sim_type) and do:
    # * read simulated map
    # * load map into timestreams, apply preprocessing
    # * apply mapmaking
    for task_element in local_mpi_list:
        patch, freq_channel, obs_id, wafer = task_element
        local_task_id = local_mpi_list.index(task_element)
        logger.debug(f"Starting task: "
                     f"({patch}, {freq_channel}, {obs_id}, {wafer})")
        
        start = time.time()

        # First, check if atomic maps already exist.
        maps_exist = True
        for sim_id, sim_type in product(sim_ids, sim_types):
            # Path to unfiltered simulation
            map_fname = sim_string_format.format(
                sim_id=sim_id,
                sim_type=sim_type,
                freq_channel=freq_channel
            )

            for split_label in intra_obs_splits:
                if (patch, freq_channel, obs_id, wafer) in atomic_metadata[split_label]:  # noqa

                    # Saving filtered atomics to disk
                    atomic_fname = map_fname.split("/")[-1].replace(
                        mfmt,
                        f"_{obs_id}_{wafer}_{split_label}{mfmt}"
                    )

                    dir_key = sim_id if sim_id is not None else sim_type
                    f_wmap = atomics_dir[patch, freq_channel][dir_key]
                    f_wmap += f"/{atomic_fname.replace(mfmt, '_wmap' + mfmt)}"
                    f_w = f_wmap.replace('_wmap' + mfmt, '_weights' + mfmt)

                    if not (os.path.isfile(f_wmap) and os.path.isfile(f_w)):
                        maps_exist = False
                else:
                    maps_exist = False

        # If they exist and we don't overwrite, skip this atomic.
        if maps_exist and not args.filtering.overwrite_atomics:
            logger.info(
                f"Map exists: ({patch}, {freq_channel}, {obs_id}, {wafer})"
                f" to filter sims {sim_ids}, {sim_types}"
            )
            continue

        # Get axis manager metadata for the given obs
        dets = {"wafer_slot": wafer, "wafer.bandpass": freq_channel}

        try:
            meta = ctx_proc.get_meta(obs_id=obs_id, dets=dets)
        except loader.LoaderError:
            logger.warning(f"NO METADATA: "
                           f"({patch}, {freq_channel}, {obs_id}, {wafer})")
            continue
        except OSError as err:
            logger.warning(f"{err}: "
                           f"({patch}, {freq_channel}, {obs_id}, {wafer})")
            continue

        # Focal plane thinning
        if args.filtering.fp_thin is not None:
            fp_thin = int(args.filtering.fp_thin)
            thinned = [
                m for im, m in enumerate(meta.dets.vals)
                if im % fp_thin == 0
            ]
            meta.restrict("dets", thinned)

        # Process data here to have t2p leakage template
        # It will stop before each step with the
        # use_data_aman flag in the preprocess config
        # files and store the AxisManager in the data_aman
        # dict.
        try:
            data_aman = pp_util.multilayer_load_and_preprocess(
                obs_id,
                configs_init,
                configs_proc,
                meta=meta,
                logger=logger,
                stop_for_sims=True,
                ignore_cfg_check=True
            )
        # After focal plane thinning, the data AxisManager might not have any
        # detectors left, resulting in one of several errors caught below.
        # TODO: We should account for those directly in sotodlib. 
        except loader.LoaderError:
            logger.warning(f"NO METADATA: "
                           f"({patch}, {freq_channel}, {obs_id}, {wafer})")
            continue
        except OSError as err:
            logger.warning(f"{err}: "
                           f"({patch}, {freq_channel}, {obs_id}, {wafer})")
            continue
        except IndexError:
            logger.warning(f"NO DETECTORS LEFT AFTER RESTRICTING: "
                           f"({patch}, {freq_channel}, {obs_id}, {wafer})")
            continue

        for sim_id, sim_type in product(sim_ids, sim_types):

            # Path to unfiltered simulation
            map_fname = sim_string_format.format(
                sim_id=sim_id,
                sim_type=sim_type,
                freq_channel=freq_channel
            )
            map_file = f"{sim_dir}/{map_fname}"

            logger.debug(f"Loading ({patch}, {freq_channel}, {obs_id}, {wafer})"
                         f" to filter {sim_type}, sim {sim_id}")
            start0 = time.time()

            logger.debug(f"Loading {pix_type} map: {map_file}")
            sim = read_map(map_file, pix_type, fields_hp=[0, 1, 2])

            try:
                aman = pp_util.multilayer_load_and_preprocess_sim(
                    obs_id,
                    configs_init=configs_init,
                    configs_proc=configs_proc,
                    sim_map=sim,
                    meta=meta,
                    logger=logger,
                    ignore_cfg_check=True,
                    data_amans=data_aman
                )
            # After focal plane thinning, the sim AxisManager might not have
            # any detectors, resulting in one of several errors caught below.
            # TODO: We should account for those directly in sotodlib. 
            except loader.LoaderError:
                logger.warning(
                    "METADATA MISSING: "
                    f"({patch}, {freq_channel}, {obs_id}, {wafer}) "
                    "SKIPPING."
                )
                continue
            except (OSError, KeyError) as err:
                logger.warning(
                    f"{err} "
                    f"({patch}, {freq_channel}, {obs_id}, {wafer}) "
                    "SKIPPING."
                )
                continue

            if aman is None:
                logger.warning(
                    "No detectors left in this atomic."
                    f"({patch}, {freq_channel}, {obs_id}, {wafer}) "
                )
                continue
            if aman.dets.count <= 1:
                logger.warning(
                    "No detectors left in this atomic."
                    f"({patch}, {freq_channel}, {obs_id}, {wafer}) "
                )
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
                    atomic_fname = map_fname.split("/")[-1].replace(
                        mfmt,
                        f"_{obs_id}_{wafer}_{split_label}{mfmt}"
                    )

                    dir_key = sim_id if sim_id is not None else sim_type
                    f_wmap = atomics_dir[patch, freq_channel][dir_key]
                    f_wmap += f"/{atomic_fname.replace(mfmt, '_wmap' + mfmt)}"
                    f_w = f_wmap.replace('_wmap' + mfmt, '_weights' + mfmt)

                    write_map(f_wmap, wmap, pix_type=pix_type, dtype=np.float32, nest=True)
                    write_map(f_w, w, pix_type=pix_type, dtype=np.float32, nest=True)
            end0 = time.time()
            logger.debug(f"Filtered in {end0 - start0:.1f} seconds: "
                         f"{sim_type}, sim {sim_id} with setup "
                         f"({patch}, {freq_channel}, {obs_id}, {wafer})")
        logger.debug(f"Processed {len(sim_ids)} simulations for "
                     f"({patch}, {freq_channel}, {obs_id}, {wafer}) in "
                     f"{time.time() - start:.1f} seconds.")
        logger.info(f"Done: {local_task_id+1}/{len(local_mpi_list)} "
                    f"for rank {rank}.")
        
    comm.Barrier()
    if rank == 0:
        end = time.time()
        print(f"Filtering completed in (wall time) {int(end - start)}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, help="yaml file with configuration."
    )
    parser.add_argument(
        "--sim_ids", type=str, default=None,
        help="Simulations to be processed, in format [first],[last]."
             "Overwrites the yaml file configs."
    )
    args = parser.parse_args()
    config = Cfg.from_yaml(args.config_file)
    if args.sim_ids is not None:
        config.filtering.update(sim_ids=args.sim_ids)

    main(config)
