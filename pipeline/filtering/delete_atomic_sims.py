import numpy as np
import argparse
import shutil
import os
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
    num_sims = args.num_sims if args.num_sims is not None else len(sim_ids)
    if len(sim_ids) != num_sims:
        raise ValueError("Incompatible number of sims between config and parser")  # noqa
    id_sim_batch = args.id_sim_batch
    num_sim_batch = args.num_sim_batch if args.num_sim_batch is not None else len(sim_ids)  # noqa
    sim_ids = [
        sim_ids[sim_id]
        for sim_id in range(id_sim_batch*num_sim_batch,
                            min(num_sims, (id_sim_batch+1)*num_sim_batch))
    ]

    logger.info("Checking coadded sims.")
    inter_obs_splits = args.inter_obs_splits
    intra_obs_splits = args.intra_obs_splits
    split_labels_all = ["science"] + intra_obs_splits + inter_obs_splits
    split_labels_all = list(dict.fromkeys(split_labels_all))
    suffixes = [".fits", "_weights.fits"]

    maps_absent = []
    for sim_id, patch, freq_channel in product(sim_ids, patches, freq_channels):  # noqa
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
    if any(maps_absent):
        raise FileNotFoundError(f"{len(maps_absent)} files missing.")
    logger.info("All bundles are present.")

    logger.debug(f"Deleting sim_ids {sim_ids}.")

    for sim_id, patch, freq_channel in product(sim_ids, patches, freq_channels):  # noqa
        map_dir = atomic_sim_dir.format(
            patch=patch,
            freq_channel=freq_labels[freq_channel],
            sim_id=sim_id
        )
        if os.path.isdir(map_dir) and args.remove_atomics:
            shutil.rmtree(map_dir)
            logger.info(f"Deleting {map_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, help="yaml file with configuration."
    )
    parser.add_argument(
        "--id_sim_batch", type=int, default=0,
        help="Number of simulations to be processed in parallel."
    )
    parser.add_argument(
        "--num_sim_batch", type=int, default=None,
        help="Number of batches of simulations to be processed in parallel."
    )
    parser.add_argument(
        "--num_sims", type=int, default=None,
        help="Total number of simulations to be processed."
    )

    args = parser.parse_args()
    config = fu.Cfg.from_yaml(args.config_file)
    config.update(vars(args))

    main(config)
