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
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'filtering'))
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
        raise ValueError("Unknown pixel type, must be 'car' or 'hp'.")
    if "{patch" not in args.bundle_db:
        raise ValueError("bundle_db does not have \
                         required placeholder 'patch'")
    for required_tag in ["{sim_id", "{sim_type"]:
        if required_tag not in args.sim_string_format:
            raise ValueError(f"sim_string_format does not have \
                             required placeholder '{required_tag}'")
    for required_tag in ["{patch", "{freq_channel"]:
        if required_tag not in args.output_dir:
            raise ValueError(f"output_dir does not have \
                             required placeholder '{required_tag}'")
    for required_tag in ["sim_id", "{patch", "{freq_channel"]:
        if required_tag not in args.atomic_sim_dir:
            raise ValueError(f"atomic_sim_dir does not have \
                             required placeholder '{required_tag}'")

    # MPI related initialization
    rank, size, comm = mpi.init(True)

    # Initialize the logger
    logger = pp_util.init_logger("benchmark", verbosity=3)

    # Ensure that freq_channels for the metadata follow the "f090" convention.
    # We keep the original labels in a dict called freq_labels.
    freq_labels = {}
    for f in args.freq_channels:  # If these don't contain the "f", add
        freq_channel = f"f{f}" if "f" not in f else f
        freq_labels[freq_channel] = f  # dict values are the original labels
    freq_channels = list(freq_labels.keys())

    # Input directory
    atomic_sim_dir = args.atomic_sim_dir

    # Output directories
    patches = args.patches
    out_dirs = {
        (patch, freq_channel):
        args.output_dir.format(
            patch=patch, freq_channel=freq_labels[freq_channel]
        )
        for patch, freq_channel in product(patches, freq_channels)
    }
    coadded_dirs = {key: f"{out_dir}/coadded_sims"
                    for key, out_dir in out_dirs.items()}
    plot_dirs = {key: f"{out_dir}/plots" for key, out_dir in out_dirs.items()}

    for key in out_dirs:
        os.makedirs(coadded_dirs[key], exist_ok=True)
        os.makedirs(plot_dirs[key], exist_ok=True)

    # Sim related arguments
    sim_types = args.sim_types
    sim_string_format = args.sim_string_format
    sim_ids = args.sim_ids
    if isinstance(sim_ids, str):
        if "," in sim_ids:
            id_min, id_max = sim_ids.split(",")
            sim_ids = np.arange(int(id_min), int(id_max)+1)
        else:
            sim_ids = np.array([int(sim_ids)])
    else:
        raise ValueError("Argument 'sim_ids' has the wrong format")
    logger.debug(f"Processing sim_ids {sim_ids} in parallel.")

    # Pixelization arguments
    pix_type = args.pix_type
    if pix_type == "hp":
        mfmt = ".fits"  # TODO: test fits.gz for HEALPix
        car_map_template = None
    elif pix_type == "car":
        mfmt = ".fits"
        car_map_template = args.car_map_template

    # Databases
    atom_db = args.atomic_db
    bundle_dbs = {patch: args.bundle_db.format(patch=patch)
                  for patch in patches}

    # Bundle query arguments
    inter_obs_splits = args.inter_obs_splits
    if inter_obs_splits is None:
        inter_obs_splits = []
    if isinstance(inter_obs_splits, str):
        if "," in inter_obs_splits:
            inter_obs_splits = inter_obs_splits.split(",")
        else:
            inter_obs_splits = [inter_obs_splits]

    # Load all data from bundle dbs without filtering
    bundle_id = args.bundle_id
    bundles = {
        patch:
        coord.BundleCoordinator.from_dbfile(
            bundle_dbs[patch], bundle_id=bundle_id
        ) for patch in patches
    }

    # Gather all split labels of atomics to be coadded
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
        (patch, inter_obs_split, None):
        bundles[patch].get_ctimes(
            bundle_id=bundle_id, null_prop_val=inter_obs_split
        ) for inter_obs_split in inter_obs_splits
        for patch in patches
    }
    # Add the science split without atomic batches
    for patch in patches:
        ctimes[patch, "science", None] = bundles[patch].get_ctimes(
            bundle_id=bundle_id
        )

    # Randomly split science ctimes into batches (optional)
    nbatches = args.nbatch_atomics
    if nbatches is not None:
        nbatches_dict = {}
        for patch in patches:
            # Limit number of batches to one half the number of ctimes
            if nbatches > len(ctimes[patch, "science", None]) // 2:
                nbatches_dict[patch] = len(ctimes[patch, "science", None]) // 2
            # Must have at least two batches
            if nbatches < 2:
                nbatches_dict[patch] = None
    batches = {}
    for patch in patches:
        batches[patch] = [None]
        if nbatches_dict[patch] is None:
            pass
        elif nbatches_dict[patch] > 1:
            nbatch = nbatches_dict[patch]
            batches[patch] = range(nbatch)
            nctimes = len(ctimes[patch, 'science', None])
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
    for patch in patches:
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
    query_restrict = args.query_restrict

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
    
    # --------------------------------------------
    # Build wafer lists per (patch, freq_channel, ib)
    # Use the wafers that appear in SCIENCE atomics for that batch,
    # so wafer coadds are consistent with the science selection.
    # --------------------------------------------
    wafers_by_pfb = None
    if args.coadd_wafers:
        wafers_by_pfb = {}
        for patch, freq_channel in product(patches, freq_channels):
            for ib in batches[patch]:
                wafers = sorted(list(set([
                    wafer for (_, wafer) in atomic_metadata[patch, freq_channel, "science", ib]
                ])))
                # If user provided a wafer list, restrict to it
                wafer_sel = getattr(args, "wafer", None)  # can be None, str, list
                if isinstance(wafer_sel, str):
                    wafer_sel = [wafer_sel]
                if wafer_sel is not None:
                    wafers = [w for w in wafers if w in wafer_sel]
                wafers_by_pfb[(patch, freq_channel, ib)] = wafers

    split_labels_all = ["science"] + intra_obs_splits + inter_obs_splits
    split_labels_all = list(dict.fromkeys(split_labels_all))

    mpi_shared_list = []

    if not args.coadd_wafers:
        mpi_shared_list = [(patch, freq_channel, sim_id, split_label, sim_type)
                           for patch in patches
                           for freq_channel in freq_channels
                           for sim_id in sim_ids
                           for split_label in split_labels_all
                           for sim_type in sim_types]

        mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
        task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list), logger=logger)
        local_mpi_list = [mpi_shared_list[i] for i in task_ids]

        loop_over = [(patch, freq, sim, split, sim_type, ib, None)
                     for patch, freq, sim, split, sim_type in local_mpi_list
                     for ib in batches[patch]]

    else:
        # Wafer-expanded behavior
        for patch in patches:
            for freq_channel in freq_channels:
                for ib in batches[patch]:
                    for wafer in wafers_by_pfb[(patch, freq_channel, ib)]:
                        for sim_id in sim_ids:
                            for split_label in split_labels_all:
                                for sim_type in sim_types:
                                    mpi_shared_list.append(
                                        (patch, freq_channel, sim_id, split_label, sim_type, ib, wafer)
                                    )

        mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
        task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list), logger=logger)
        loop_over = [mpi_shared_list[i] for i in task_ids]

    for patch, freq_channel, sim_id, split_label, sim_type, ib, wafer in loop_over:

        map_dir = atomic_sim_dir.format(
            patch=patch,
            freq_channel=freq_labels[freq_channel],
            sim_id=sim_id
        )
        assert os.path.isdir(map_dir), map_dir

        if not ib:
            logger.info(f"Loading atomics for ({patch}, {freq_channel}, "
                        f"{split_label})"
                        f" to filter {sim_type}, {split_label}, sim {sim_id}")

        w_list, wmap_list = ([], [])
        
        def _filter_by_wafer(meta_list, wafer_sel):
            if wafer_sel is None:
                return meta_list
            return [(obs_id, w) for (obs_id, w) in meta_list if w == wafer_sel]

        if split_label == "science":
            tracemalloc.start()
            for coadd in intra_obs_pair:
                meta = _filter_by_wafer(
                    atomic_metadata[patch, freq_channel, coadd, ib], wafer
                )
                if len(meta) == 0:
                    continue
                wmap_l, w_l = fu.get_atomics_maps_list(
                    sim_id, sim_type,
                    meta,
                    freq_labels[freq_channel], map_dir, coadd,
                    sim_string_format, mfmt=mfmt, pix_type=pix_type,
                    logger=logger
                )

                wmap_list += wmap_l
                w_list += w_l
            current_gb, peak_gb = [1024**(-3) * c
                                   for c in tracemalloc.get_traced_memory()]
            logger.info("Traced Memory for 'science' (Current, Peak): "
                        f"{current_gb:.2f} GB, {peak_gb:.2f} GB")
            tracemalloc.stop()
        elif split_label in inter_obs_splits:
            for coadd in intra_obs_pair:
                meta = _filter_by_wafer(
                    atomic_metadata[patch, freq_channel, split_label, coadd, ib], wafer
                )
                if len(meta) == 0:
                    continue
                wmap_l, w_l = fu.get_atomics_maps_list(
                    sim_id, sim_type,
                    meta,
                    freq_channel, map_dir, coadd,
                    sim_string_format, mfmt=mfmt, pix_type=pix_type,
                    logger=logger
                )

                wmap_list += wmap_l
                w_list += w_l
        else:
            meta = _filter_by_wafer(
                atomic_metadata[patch, freq_channel, split_label, ib], wafer
            )
            if len(meta) == 0:
                continue
            wmap_list, w_list = fu.get_atomics_maps_list(
                sim_id, sim_type,
                meta,
                freq_channel, map_dir, split_label,
                sim_string_format, mfmt=mfmt, pix_type=pix_type,
                logger=logger
            )


        if not ib:
            logger.info(f"Coadding atomics for ({patch}, {freq_channel}, "
                        f"{split_label})"
                        f" to filter {sim_type}, sim {sim_id}")

        if len(wmap_list) == 0:
            if not ib:
                logger.info(
                    f"Skipping (no atomics) for wafer={wafer} "
                    f"({patch}, {freq_channel}, {split_label}), sim={sim_id}, type={sim_type}"
                )
            continue

        map_filtered, weights = bu.coadd_maps(
            wmap_list, w_list, pix_type=pix_type,
            car_template_map=car_map_template
        )
        out_fname = sim_string_format.format(
            sim_id=sim_id,
            sim_type=sim_type,
            freq_channel=freq_labels[freq_channel]
        ).split("/")[-1]
        batch_label = "" if ib is None else f"_batch{ib}of{nbatches}"
        wafer_tag = "" if wafer is None else f"_wafer{wafer}"
        out_fname = out_fname.replace(
            ".fits",
            f"_bundle{bundle_id}_{freq_channel}_{split_label}{wafer_tag}{batch_label}_filtered.fits"
        )

        fu.save_and_plot_map(
            map_filtered, out_fname,
            coadded_dirs[patch, freq_channel],
            plot_dirs[patch, freq_channel],
            pix_type=pix_type
        )
        fu.save_and_plot_map(
            weights, out_fname.replace(".fits", "_weights.fits"),
            coadded_dirs[patch, freq_channel],
            plot_dirs[patch, freq_channel],
            pix_type=pix_type, do_plot=False
        )
    comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, help="yaml file with configuration."
    )
    #parser.add_argument(
    #    "--sim_ids", type=str, default=0,
    #    help="Simulations to be processed, in format [first],[last]."
    #         "Overwrites the yaml file configs."
    #)
    parser.add_argument(
        "--sim_ids", type=str, default=None,
        help="Simulations to be processed, in format 'first,last'. Overrides YAML if provided."
    )

    args = parser.parse_args()
    config = fu.Cfg.from_yaml(args.config_file)
    #config.update(vars(args))
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(overrides)
    
    # wafer in YAML implies wafer coadds
    if hasattr(config, "wafer") and config.wafer:
        config.coadd_wafers = True
    else:
        config.coadd_wafers = False
        config.wafer = None


    main(config)
