import numpy as np
import argparse
import sqlite3
import os
import shutil
import sys
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
import mpi_utils as mpi  # noqa
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
    elif pix_type == "car":
        mfmt = ".fits"

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
        if batches[patch] != [None]:
            logger.warning(f"{patch}: setting nbatch_atomics to 1.")
            batches[patch] = [None]

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

    relevant_splits = ["science"] + intra_obs_splits
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
    atomics_list = [(patch, freq_channel, split_label, sim_type)
                    for patch in patches
                    for freq_channel in freq_channels
                    for split_label in split_labels_all
                    for sim_type in sim_types]

    loop_over = [(patch, freq, split, sim_type, ib)
                 for ib in batches[patch]
                 for patch, freq, split, sim_type in atomics_list]

    for sim_id in sim_ids:
        natomics_ideal = 0
        natomics_real = 0
        for patch, freq_channel, split_label, sim_type, ib in loop_over:
            map_dir = atomic_sim_dir.format(
                patch=patch,
                freq_channel=freq_labels[freq_channel],
                sim_id=sim_id
            )
            if not os.path.isdir(map_dir):
                logger.warning(f"Directory is missing: {map_dir}")
            elif args.force_delete:
                logger.info(f"Deleting {map_dir}")
                shutil.rmtree(map_dir)

            if split_label == "science":
                for coadd in intra_obs_pair:
                    num_ideal, num_real = fu.get_atomics_maps_list(
                        sim_id, sim_type,
                        atomic_metadata[patch, freq_channel, coadd, ib],
                        freq_labels[freq_channel], map_dir, coadd,
                        sim_string_format, mfmt=mfmt, pix_type=pix_type,
                        logger=logger, file_stats_only=True
                    )
                    natomics_ideal += num_ideal
                    natomics_real += num_real
            elif split_label in inter_obs_splits:
                for coadd in intra_obs_pair:
                    num_ideal, num_real = fu.get_atomics_maps_list(
                        sim_id, sim_type,
                        atomic_metadata[patch, freq_channel, split_label, coadd, ib],  # noqa
                        freq_channel, map_dir, coadd,
                        sim_string_format, mfmt=mfmt, pix_type=pix_type,
                        logger=logger, file_stats_only=True
                    )
                    natomics_ideal += num_ideal
                    natomics_real += num_real
            else:
                num_ideal, num_real = fu.get_atomics_maps_list(
                    sim_id, sim_type,
                    atomic_metadata[patch, freq_channel, split_label, ib],
                    freq_channel, map_dir, split_label,
                    sim_string_format, mfmt=mfmt, pix_type=pix_type,
                    logger=logger, file_stats_only=True
                )
                natomics_ideal += num_ideal
                natomics_real += num_real

        logger.info(f"Sim ID {sim_id}: {natomics_real} of {natomics_ideal}"
                    f" present ({natomics_real/natomics_ideal*100:.1f} %)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, help="yaml file with configuration."
    )
    parser.add_argument(
        "--sim_ids", type=str, default=0,
        help="Simulations to be processed, in format [first],[last]."
             "Overwrites the yaml file configs."
    )
    parser.add_argument(
        "--force_delete", action="store_true",
        help="Force deletion of all atomics found on disk."
    )
    args = parser.parse_args()
    config = fu.Cfg.from_yaml(args.config_file)
    config.update(vars(args))

    main(config)
