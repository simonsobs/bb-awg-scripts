"""
Based on coadd_filtered_ext.py, 
adapted to run on the external data.
"""
import numpy as np
import argparse
import sqlite3
import os
import sys
import tracemalloc
from itertools import product

import sotodlib.preprocess.preprocess_util as pp_util

# TODO: Make it an actual module
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bundling'))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'misc'))
)
import mpi_utils as mpi # noqa
import bundling_utils as bu  # noqa
import filtering_utils as fu  # noqa
import coordinator as coord  # noqa


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

    # Output directories
    out_dir = args.output_dir
    coadded_dir = out_dir + "/coadded_maps"
    plot_dir = out_dir + "/plots"
    os.makedirs(coadded_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Path to atomic maps
    atomic_sim_dir = args.atomic_sim_dir

    # Databases
    atom_db = args.atomic_db
    bundle_db = args.bundle_db

    # Sim related arguments
    sim_string_format = args.sim_string_format
    # sim_ids = args.sim_ids
    # if isinstance(sim_ids, str):
    #     if "," in sim_ids:
    #         id_min, id_max = sim_ids.split(",")
    #         sim_ids = np.arange(int(id_min), int(id_max)+1)
    #     else:
    #         sim_ids = np.array([int(sim_ids)])

    # Pixelization arguments
    pix_type = args.pix_type
    if pix_type == "hp":
        mfmt = ".fits"  # TODO: test fits.gz for HEALPix
        car_map_template = None
    elif pix_type == "car":
        mfmt = ".fits"
        car_map_template = args.car_map_template

    # Bundle query arguments
    freq_channel = args.freq_channel
    inter_obs_splits = args.inter_obs_splits
    if inter_obs_splits is None:
        inter_obs_splits = []
    if isinstance(inter_obs_splits, str):
        if "," in inter_obs_splits:
            inter_obs_splits = inter_obs_splits.split(",")
        else:
            inter_obs_splits = [inter_obs_splits]

    # Load all data from bundle db without filtering
    bundle_id = args.bundle_id
    bundle_db = coord.BundleCoordinator.from_dbfile(
        bundle_db, bundle_id=bundle_id
    )

    # Gather all split labels of atomics to be coadded.
    intra_obs_splits = args.intra_obs_splits
    intra_obs_pair = args.intra_obs_pair
    if intra_obs_splits is not None:
        if isinstance(intra_obs_splits, str):
            if "," not in intra_obs_splits:
                intra_obs_splits = [intra_obs_splits]
            else:
                intra_obs_splits = (intra_obs_splits).split(",")
    if intra_obs_pair is not None:
        if isinstance(intra_obs_pair, str):
            if "," not in intra_obs_pair:
                raise ValueError("You must pass a comma-separated string list "
                                 "to 'intra_obs_pair'.")
            else:
                intra_obs_pair = intra_obs_pair.split(",")
    if ((intra_obs_splits, intra_obs_pair) == (None, None)):
        raise ValueError("You must pass at least one of the two: "
                         "'intra_obs_pair' or 'intra_obs_splits'.")

    logger.info(f"Split labels to coadd individually: {intra_obs_splits}")
    logger.info(f"Split labels to coadd together: {intra_obs_pair}")

    # Extract list of ctimes from bundle database for the given
    # bundle_id and without atomic batches - inter obs null label
    ctimes = {
        (inter_obs_split, None): bundle_db.get_ctimes(
            bundle_id=bundle_id, null_prop_val=inter_obs_split
        ) for inter_obs_split in inter_obs_splits
    }
    # Add the science split without atomic batches
    ctimes["science", None] = bundle_db.get_ctimes(bundle_id=bundle_id)

    # Randomly split science ctimes into batches
    nbatches = args.nbatch_atomics
    batches = range(nbatches) if nbatches > 1 else [None]
    if nbatches is not None:
        # Limit number of batches to one half the number of ctimes
        if nbatches > len(ctimes["science", None]) // 2:
            nbatches = len(ctimes["science", None]) // 2
        # Must have at least two batches
        if nbatches < 2:
            nbatches = None

    if nbatches is not None:
        logger.info(
            f"Splitting atomics into {nbatches} random batches with "
            f"{len(ctimes['science', None]) // nbatches} ctimes in each."
        )
        idx_rand = np.random.permutation(range(len(ctimes["science", None])))
        for ib in batches:
            ctimes["science", ib] = [ctimes["science", None][i]
                                     for i in idx_rand
                                     if (i+ib) % nbatches]

    # Restrict the inter-obs null splits to the ctimes of the "science" split
    for inter_obs_split, ib in product(inter_obs_splits, batches):
        ctimes[inter_obs_split, ib] = [ct
                                       for ct in ctimes[inter_obs_split, None]
                                       if ct in ctimes["science", ib]]

    # Connect the the atomic map DB
    db_con = sqlite3.connect(atom_db)
    db_cur = db_con.cursor()
    query_restrict = args.query_restrict
    atomic_metadata = {}

    # Query all atomics used for science, filtering ctimes
    for ib in batches:
        query = fu.get_query_atomics(
            freq_channel, ctimes["science", ib], split_label="science",
            query_restrict=query_restrict
        )
        res = db_cur.execute(query)
        res = res.fetchall()
        atomic_metadata["science", ib] = [
            (obs_id, wafer) for obs_id, wafer in res
        ]

    # Query all atomics used for intra-obs splits
    # filtering ctimes and split labels
    for intra_obs_split, ib in product(intra_obs_splits, batches):
        query = fu.get_query_atomics(
            freq_channel, ctimes["science", ib], split_label=intra_obs_split,
            query_restrict=query_restrict
        )
        res = db_cur.execute(query)
        res = res.fetchall()
        atomic_metadata[intra_obs_split, ib] = [
            (obs_id, wafer) for obs_id, wafer in res
            if (obs_id, wafer) in atomic_metadata["science", ib]
        ]

    # Query all atomics used for inter-obs splits
    # filtering ctimes w.r.t to the null prop considered
    # for the two intra-obs splits to be coadded
    if len(intra_obs_pair) != 0:
        for inter_obs_split, intra_obs_split, ib in product(inter_obs_splits,
                                                            intra_obs_pair,
                                                            batches):
            query = fu.get_query_atomics(
                freq_channel, ctimes[inter_obs_split, ib],
                split_label=intra_obs_split
            )
            res = db_cur.execute(query)
            res = res.fetchall()
            atomic_metadata[inter_obs_split, intra_obs_split, ib] = [
                (obs_id, wafer) for obs_id, wafer in res
                if (obs_id, wafer) in atomic_metadata["science", ib]
            ]
    db_con.close()

    split_labels_all = ["science"] + intra_obs_splits + inter_obs_splits
    split_labels_all = list(dict.fromkeys(split_labels_all))  # no duplicates
    mpi_shared_list = [(split_label, pure_type)  #sim_id
                    #    for sim_id in sim_ids
                       for split_label in split_labels_all
                       for pure_type in [f"pure{i}" for i in "TEB"]]

    # Every rank must have the same shared list
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list),
                                    logger=logger)
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]
    loop_over = [(split, pure, ib)   #sim, 
                 for split, pure in local_mpi_list  # sim,
                 for ib in batches]

    for split_label, pure_type, ib in loop_over:  #sim_id
        map_dir = atomic_sim_dir  #.format(sim_id=sim_id)

        assert os.path.isdir(map_dir), map_dir

        if not ib:
            logger.info(
                f"Loading atomics for sim {split_label}, "
                f"{pure_type}"
            )

        w_list, wmap_list = ([], [])

        if split_label == "science":
            tracemalloc.start()
            for coadd in intra_obs_pair:
                wmap_l, w_l = fu.get_atomics_maps_list(
                    None, pure_type, atomic_metadata[coadd, ib],
                    freq_channel, map_dir, coadd,
                    sim_string_format, mfmt=mfmt, pix_type=pix_type,
                    logger=logger
                )
                wmap_list += wmap_l
                w_list += w_l
            current_gb, peak_gb = [1024**(-3) * c
                                   for c in tracemalloc.get_traced_memory()]
            logger.info("Traced Memory for 'science' (Current, Peak): "
                        f"({current_gb:.2f} GB, {peak_gb:.2f} GB")
            tracemalloc.stop()
        elif split_label in inter_obs_splits:
            for coadd in intra_obs_pair:
                wmap_l, w_l = fu.get_atomics_maps_list(
                    None, pure_type, atomic_metadata[split_label, coadd, ib],
                    freq_channel, map_dir, coadd,
                    sim_string_format, mfmt=mfmt, pix_type=pix_type,
                    logger=logger
                )
                wmap_list += wmap_l
                w_list += w_l
        else:
            wmap_list, w_list = fu.get_atomics_maps_list(
                None, pure_type, atomic_metadata[split_label, ib],
                freq_channel, map_dir, split_label,
                sim_string_format, mfmt=mfmt, pix_type=pix_type,
                logger=logger
            )

        if not ib:
            logger.info(
                f"Coadding atomics for {split_label}"  # sim {sim_id:04d} and
            )

        map_filtered, weights = bu.coadd_maps(
            wmap_list, w_list, pix_type=pix_type,
            car_template_map=car_map_template
        )
        out_fname = sim_string_format #.format(sim_id=sim_id,
                                            #  pure_type=pure_type)
        batch_label = "" if ib is None else f"_batch{ib}of{nbatches}"
        out_fname = out_fname.replace(
            ".fits",
            f"_bundle{bundle_id}_{freq_channel}_{split_label}{batch_label}_filtered.fits"  # noqa
        )
        fu.save_and_plot_map(
            map_filtered, out_fname, coadded_dir, plot_dir,
            pix_type=pix_type
        )
        fu.save_and_plot_map(
            weights, out_fname.replace(".fits", "_weights.fits"),
            coadded_dir, plot_dir, pix_type=pix_type, do_plot=False
        )
    comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, help="yaml file with configuration."
    )

    args = parser.parse_args()
    config = fu.Cfg.from_yaml(args.config_file)
    main(config)
