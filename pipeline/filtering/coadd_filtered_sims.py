import numpy as np
import argparse
import sqlite3
import os
import sys
import time
import tracemalloc
from itertools import product

import sotodlib.preprocess.preprocess_util as pp_util

# TODO: Make it an actual module
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bundling'))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'filtering'))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'misc'))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
import mpi_utils as mpi # noqa
import bundling_utils as bu  # noqa
import filtering_utils as fu  # noqa
import coordinator as coord  # noqa
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
    atomic_sim_dir = args.filtering.atomic_sim_dir
    freq_channels = np.atleast_1d(args.freq_channel)
    patches = np.atleast_1d(args.patch)
    atom_db = args.atomic_db

    # Output directories
    out_dirs = {}
    for labels in product(patches, freq_channels, sim_types):
        out_dirs[labels] = args.filtering.output_dir_filtering.format(
            patch=labels[0], freq_channel=labels[1], sim_type=labels[2])

    if args.filtering.coadded_dirs is None:
        coadded_dir = f"{args.filtering.output_dir_filtering}/coadded_sims"
    else:
        coadded_dir = args.filtering.coadded_dirs
    coadded_dirs = {
        key: coadded_dir.format(patch=key[0], freq_channel=key[1], sim_type=key[2])
        for key in out_dirs
    }
    plot_dirs = {key: f"{out_dir}/plots" for key, out_dir in out_dirs.items()}

    for key in out_dirs:
        os.makedirs(coadded_dirs[key], exist_ok=True)
        os.makedirs(plot_dirs[key], exist_ok=True)

    # Pixelization arguments
    pix_type, mfmt, car_map_template, _, _ = fu.get_pix_type_args(args)

    # Bundle query arguments
    inter_obs_splits = args.inter_obs_splits
    if inter_obs_splits is None:
        inter_obs_splits = []
    if isinstance(inter_obs_splits, str):
        if "," in inter_obs_splits:
            inter_obs_splits = inter_obs_splits.split(",")
        else:
            inter_obs_splits = [inter_obs_splits]

    # Gather all split labels of atomics to be coadded
    if args.intra_obs_splits in [None, [], [None], "None"]:
        intra_obs_splits = []
    else:
        intra_obs_splits = args.intra_obs_splits
    if args.intra_obs_pair in [None, [], [None], "None"]:
        intra_obs_pair = []
    else:
        intra_obs_pair = args.intra_obs_pair
    if ((intra_obs_splits, intra_obs_pair) == ([], [])):
        raise ValueError("You must pass at least one of the two: "
                         "'intra_obs_pair' or 'intra_obs_splits'.")
    if intra_obs_splits != []:
        if isinstance(intra_obs_splits, str):
            if "," not in intra_obs_splits:
                intra_obs_splits = [intra_obs_splits]
            else:
                intra_obs_splits = (intra_obs_splits).split(",")
    if intra_obs_pair != []:
        if isinstance(intra_obs_pair, str):
            if "," not in intra_obs_pair:
                raise ValueError("You must pass a comma-separated string list "
                                 "to 'intra_obs_pair'.")
            else:
                intra_obs_pair = intra_obs_pair.split(",")

    if rank == 0:
        logger.info(f"Split labels to coadd individually: {intra_obs_splits}")
        logger.info(f"Split labels to coadd together: {intra_obs_pair}")

    # Extract list of ctimes from bundle database for the given
    # bundle_id and without atomic batches - inter obs null label
    bundle_id = args.filtering.bundle_id
    bundles = {}
    ctimes = {}
    # Randomly split science ctimes into batches (optional)
    nbatches = args.filtering.nbatch_atomics
    if nbatches is None:
        nbatches = 1
    nbatches_dict = {}
    batches = {}
    for patch in patches:
        args_patch = args.child(patch=patch)
        bundle = coord.BundleCoordinator.from_dbfile(
            args_patch.bundle_db_full, bundle_id=bundle_id)
        bundles[patch] = bundle

        for inter_obs_split in inter_obs_splits:
            ctime = bundle.get_ctimes(bundle_id=bundle_id, null_prop_val=inter_obs_split)
            ctimes[(patch, inter_obs_split, None)] = ctime

        ctimes[patch, "science", None] = bundle.get_ctimes(bundle_id=bundle_id)

        # Limit number of batches to one half the number of ctimes
        if nbatches > len(ctimes[patch, "science", None]) // 2:
            nbatches_dict[patch] = len(ctimes[patch, "science", None]) // 2
        # Must have at least two batches
        if nbatches < 2:
            nbatches_dict[patch] = None
        nbatch = nbatches_dict[patch]
        batches[patch] = [None]
        if nbatch is None:
            pass
        elif nbatch > 1:
            batches[patch] = range(nbatch)
            nctimes = len(ctimes[patch, "science", None])
            if rank == 0:
                logger.info(
                    f"{patch}: splitting atomics into {nbatch} random "
                    f"batches with {nctimes // nbatch} ctimes in each."
                )
            idx_rand = np.random.permutation(range(nctimes))
            for ib in batches[patch]:
                ctimes[patch, "science", ib] = [
                    ctimes[patch, "science", None][i]
                    for i in idx_rand if (i+ib) % nbatch
                ]

    # Restrict the inter-obs null splits to the ctimes of the "science" split
        for inter_obs_split, ib in product(inter_obs_splits, batches[patch]):
            ctimes[patch, inter_obs_split, ib] = [
                ct
                for ct in ctimes[patch, inter_obs_split, None]
                if ct in ctimes[patch, "science", ib]
            ]

    # Connect the the atomic map DB
    db_con = sqlite3.connect(atom_db)
    db_cur = db_con.cursor()

    # TODO: check if query_restrict is channel- or patch-specific
    query_restrict = args.query_restrict  # Could add patch query but shouldn't be necessary

    relevant_splits = list(set(["science"] + intra_obs_splits + intra_obs_pair))  # noqa
    queries = {
        (patch, freq_channel, split_label, ib): fu.get_query_atomics(
            freq_channel, ctimes[patch, "science", ib],
            split_label=split_label, query_restrict=query_restrict
        )
        for ib in batches[patch]
        for patch, freq_channel, split_label in product(patches,
                                                        freq_channels,
                                                        relevant_splits)
    }
    atomic_metadata = {key: [] for key in queries}

    # Query all atomics used for science, filtering ctimes
    for (patch, freq_channel, split_label, ib), query in queries.items():
        if split_label == "science":
            res = db_cur.execute(query)
            res = res.fetchall()
            atomic_metadata[patch, freq_channel, "science", ib] = [
                (obs_id, wafer) for obs_id, wafer in res
            ]

    # Query all atomics used for intra-obs splits
    # filtering ctimes and split labels
    for (patch, freq_channel, split_label, ib), query in queries.items():
        if split_label != "science":
            res = db_cur.execute(query)
            res = res.fetchall()
            atomic_metadata[patch,
                            freq_channel,
                            split_label,
                            ib] = [
                (obs_id, wafer) for obs_id, wafer in res
                if (obs_id, wafer) in atomic_metadata[patch,
                                                      freq_channel,
                                                      "science",
                                                      ib]
            ]

    # Query all atomics used for inter-obs splits
    # filtering ctimes w.r.t to the null prop considered
    # for the two intra-obs splits to be coadded
    if len(intra_obs_pair) != 0:
        for patch, freq_channel in product(patches, freq_channels):
            for inter_obs_split, intra_obs_split, ib in product(inter_obs_splits,  # noqa
                                                                intra_obs_pair,  # noqa
                                                                batches[patch]):  # noqa
                query = fu.get_query_atomics(
                    freq_channel, ctimes[patch, inter_obs_split, ib],
                    split_label=intra_obs_split
                )
                res = db_cur.execute(query)
                res = res.fetchall()
                atomic_metadata[patch,
                                freq_channel,
                                inter_obs_split,
                                intra_obs_split,
                                ib] = [
                    (obs_id, wafer) for obs_id, wafer in res
                    if (obs_id, wafer) in atomic_metadata[patch,
                                                          freq_channel,
                                                          "science",
                                                          ib]
                ]
    db_con.close()

    split_labels_all = ["science"] + intra_obs_splits + inter_obs_splits
    split_labels_all = list(dict.fromkeys(split_labels_all))  # no duplicates
    mpi_shared_list = [(patch, freq_channel, sim_id, split_label, sim_type)
                       for patch in patches
                       for freq_channel in freq_channels
                       for sim_id in sim_ids
                       for split_label in split_labels_all
                       for sim_type in sim_types]

    # Every rank must have the same shared list
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list),
                                    logger=logger)
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]
    loop_over = [(patch, freq, sim, split, sim_type, ib)
                 for ib in batches[patch]
                 for patch, freq, sim, split, sim_type in local_mpi_list]

    for patch, freq_channel, sim_id, split_label, sim_type, ib in loop_over:
        task_element = (patch, freq_channel, sim_id, split_label, sim_type)
        local_task_id = local_mpi_list.index(task_element)
        if sim_id is None:
            map_dir = atomic_sim_dir.format(
                patch=patch,
                freq_channel=freq_channel,
                sim_type=sim_type
            )
        else:
            map_dir = atomic_sim_dir.format(
                patch=patch,
                freq_channel=freq_channel,
                sim_id=sim_id
            )
        assert os.path.isdir(map_dir), map_dir

        if not ib:
            logger.debug(f"Loading atomics for ({patch}, {freq_channel}, "
                         f"{split_label})"
                         f" to filter {sim_type}, {split_label}, sim {sim_id}")

        w_list, wmap_list = ([], [])

        if split_label == "science":
            tracemalloc.start()
            for coadd in intra_obs_pair:
                wmap_l, w_l = fu.get_atomics_maps_list(
                    sim_id, sim_type,
                    atomic_metadata[patch, freq_channel, coadd, ib],
                    freq_channel, map_dir, coadd,
                    sim_string_format, mfmt=mfmt, pix_type=pix_type,
                    logger=logger
                )
                wmap_list += wmap_l
                w_list += w_l
            current_gb, peak_gb = [1024**(-3) * c
                                   for c in tracemalloc.get_traced_memory()]
            logger.debug("Traced Memory for 'science' (Current, Peak): "
                         f"{current_gb:.2f} GB, {peak_gb:.2f} GB")
            tracemalloc.stop()
        elif split_label in inter_obs_splits:
            for coadd in intra_obs_pair:
                wmap_l, w_l = fu.get_atomics_maps_list(
                    sim_id, sim_type,
                    atomic_metadata[patch, freq_channel, split_label, coadd, ib],  # noqa
                    freq_channel, map_dir, coadd,
                    sim_string_format, mfmt=mfmt, pix_type=pix_type,
                    logger=logger
                )
                wmap_list += wmap_l
                w_list += w_l
        else:
            wmap_list, w_list = fu.get_atomics_maps_list(
                sim_id, sim_type,
                atomic_metadata[patch, freq_channel, split_label, ib],
                freq_channel, map_dir, split_label,
                sim_string_format, mfmt=mfmt, pix_type=pix_type,
                logger=logger
            )

        if not ib:
            logger.debug(f"Coadding atomics for ({patch}, {freq_channel}, "
                         f"{split_label})"
                         f" to filter {sim_type}, sim {sim_id}")

        map_filtered, weights = bu.coadd_maps(
            wmap_list, w_list, pix_type=pix_type,
            car_template_map=car_map_template
        )
        out_fname = sim_string_format.format(
            sim_id=sim_id,
            sim_type=sim_type,
            freq_channel=freq_channel
        ).split("/")[-1]
        batch_label = "" if ib is None else f"_batch{ib}of{nbatches}"
        out_fname = out_fname.replace(
            ".fits",
            f"_bundle{bundle_id}_{freq_channel}_{split_label}{batch_label}_filtered.fits"  # noqa
        )
        fu.save_and_plot_map(
            map_filtered, out_fname,
            coadded_dirs[(patch, freq_channel, sim_type)],
            plot_dirs[(patch, freq_channel, sim_type)],
            pix_type=pix_type
        )
        fu.save_and_plot_map(
            weights, out_fname.replace(".fits", "_weights.fits"),
            coadded_dirs[(patch, freq_channel, sim_type)],
            plot_dirs[(patch, freq_channel, sim_type)],
            pix_type=pix_type, do_plot=False
        )
        if ib in [None, batches[patch][-1]]:
            logger.info(f"Done: {local_task_id+1}/{len(local_mpi_list)} "
                        f"for rank {rank}.")

    # All ranks sync here after completing their tasks
    comm.barrier()

    if rank == 0:
        end = time.time()
        print(f"Coadding completed in (wall time) {int(end - start)}s.")


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
