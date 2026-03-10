import numpy as np
import argparse
import shutil
import os
import sys
import time
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
import filtering_utils as fu  # noqa


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

    # Initialize the logger
    logger = pp_util.init_logger("benchmark", verbosity=3)
    start = time.time()

    if args.nbatch_atomics > 1:
        raise NotImplementedError("Only nbatch_atomics=1 is supported.")

    # Ensure that freq_channels for the metadata follow the "f090" convention.
    # We keep the original labels in a dict called freq_labels.
    freq_labels = {}
    for f in args.freq_channels:  # If these don't contain the "f", add
        freq_channel = f"f{f}" if "f" not in f else f
        freq_labels[freq_channel] = f  # dict values are the original labels
    freq_channels = list(freq_labels.keys())

    # Sim related arguments
    out_dir = args.output_dir
    sim_types = args.sim_types
    sim_string_format = args.sim_string_format
    atomic_sim_dir = args.atomic_sim_dir
    patches = args.patches
    sim_ids = args.sim_ids
    bundle_id = args.bundle_id

    if isinstance(sim_ids, str):
        if "," in sim_ids:
            id_min, id_max = sim_ids.split(",")
            sim_ids = np.arange(int(id_min), int(id_max)+1)
        else:
            sim_ids = np.array([int(sim_ids)])
    else:
        raise ValueError("Argument 'sim_ids' has the wrong format")

    logger.info("Checking coadded sims.")
    inter_obs_splits = args.inter_obs_splits
    intra_obs_splits = args.intra_obs_splits
    split_labels_all = ["science"] + intra_obs_splits + inter_obs_splits
    split_labels_all = list(dict.fromkeys(split_labels_all))
    suffixes = [".fits", "_weights.fits"]

    maps_absent_all = []
    for sim_id, patch, freq_channel in product(sim_ids, patches, freq_channels):  # noqa
        maps_absent = []
        coadded_dir = out_dir.format(
            patch=patch,
            freq_channel=freq_labels[freq_channel]
        ) + "/coadded_sims"
        for sim_type, split, suffix in product(sim_types, split_labels_all, suffixes):  # noqa
            fname = sim_string_format.format(
                sim_id=sim_id,
                sim_type=sim_type,
                freq_channel=freq_labels[freq_channel]
            ).split("/")[-1]
            out_fname = fname.replace(
                ".fits",
                f"_bundle{bundle_id}_{freq_channel}_{split}_filtered{suffix}"
            )
            if not os.path.isfile(f"{coadded_dir}/{out_fname}"):
                maps_absent.append(out_fname)
                maps_absent_all.append(out_fname)
        if any(maps_absent):
            logger.warning(f"(sim {sim_id}, {patch}, {freq_channel}): "
                           f"{len(maps_absent)} files missing.")
        else:
            logger.debug(f"(sim {sim_id}, {patch}, {freq_channel})")

    if any(maps_absent_all) and not args.force_delete:
        raise FileNotFoundError(f"{len(maps_absent)} files missing.")
    logger.info("All bundles are present.")

    logger.debug(f"Checking sim_ids {sim_ids}.")

    for sim_id, patch, freq_channel in product(sim_ids, patches, freq_channels):  # noqa
        map_dir = atomic_sim_dir.format(
            patch=patch,
            freq_channel=freq_labels[freq_channel],
            sim_id=sim_id
        )
        if os.path.isdir(map_dir) and (args.remove_atomics
                                       or args.force_delete):
            shutil.rmtree(map_dir)
            logger.info(f"Deleting {map_dir}")

    end = time.time()
    print(f"Deletion of atomics completed in (wall time) {int(end - start)}s.")


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
        help="Force delete atomics."
    )

    args = parser.parse_args()
    config = fu.Cfg.from_yaml(args.config_file)
    config.update(vars(args))

    main(config)
