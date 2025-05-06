import argparse
import os
import sys
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pixell import enmap, enplot

from coadder import SignFlipper
from coordinator import BundleCoordinator
import bundling_utils

sys.path.append("/home/sa5705/software/bb-awg-scripts/pipeline/misc")
import mpi_utils as mpi  # noqa

def main(args):
    """
    Main function to create sign-flipped noise realizations.
    """

    args = args_dict = args.copy()  # Make sure we don't modify input
    patch = args.patch
    query_restrict = args.query_restrict
    if patch is not None:
        if query_restrict:
            query_restrict += " AND "
        if patch == "south":
            query_restrict += "(azimuth > 90 AND azimuth < 270)"
        elif patch == "north":
            query_restrict += "(azimuth < 90 OR azimuth > 270)"
        else:
            raise ValueError(f"Patch {patch} not recognized.")
    
    # Read bundle.db
    if os.path.isfile(args.bundle_db) and not args.overwrite:
        print(f"Loading from {args.bundle_db}.")
        bundle_coordinator = BundleCoordinator.from_dbfile(
            args.bundle_db,
            null_prop_val=args.null_prop_val_inter_obs
        )
    else:
        raise FileNotFoundError(f"File {args.bundle_db} does not exist.")

    if args.pix_type not in ["hp", "car"]:
        raise ValueError("Unknown pixel type, must be 'car' or 'hp'.")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    atomic_list = None
    if args.atomic_list is not None:
        atomic_list = np.load(args.atomic_list)["atomic_list"]

    car_map_template = args.car_map_template

    bundle_ids = range(args.n_bundles)
    n_sims = config.n_sims

    # Figure out split tag ahead of time
    split_intra_obs, split_inter_obs = (args.split_label_intra_obs,
                                        args.null_prop_val_inter_obs)

    if (split_intra_obs, split_inter_obs) == (None, None):
        split_tag = "science"
    elif split_inter_obs is not None:
        split_tag = split_inter_obs
    elif split_intra_obs is not None:
        split_tag = split_intra_obs
    if isinstance(split_tag, list):
        split_tag = '_'.join(split_tag)

    wafer_tag = args.wafer if args.wafer is not None else ""
    patch_tag = args.patch if args.patch is not None else ""

    # --------------------------------------------
    # Precompute all missing tasks
    missing_tasks = []
    print(split_tag)
    for bundle_id in bundle_ids:
        print("  ", bundle_id)
        for sim_id in range(n_sims):
            out_fname = os.path.join(
                out_dir,
                args.map_string_format.format(
                    split=split_tag,
                    bundle_id=bundle_id,
                    wafer=wafer_tag,
                    patch=patch_tag,
                    freq_channel=args.freq_channel
                )
            )
            out_fname = out_fname.replace("__", "_")
            out_fname = out_fname.replace("_map.fits", f"_{sim_id:04d}_map.fits")
            if not os.path.exists(out_fname) or args.overwrite:
                print(f"{sim_id:04d}") #{out_fname}")
                missing_tasks.append((bundle_id, sim_id))
    # --------------------------------------------

    n_missing = len(missing_tasks)
    print(f"{n_missing} tasks missing out of {args.n_bundles * n_sims} total.")

    if n_missing == 0:
        print("All requested simulations already exist. Nothing to do.")
        return

    # Initialize MPI
    mpi.init(True)

    # Loop over only missing tasks
    #for task_id in mpi.taskrange(len(missing_tasks)):
    for task_id in mpi.taskrange(n_missing - 1):
        bundle_id, sim_id = missing_tasks[task_id]
        print(f"Running bundle_id={bundle_id} sim_id={sim_id}")

        # Now construct the signflipper (only for needed bundle_id)
        signflipper = SignFlipper(
            atomic_db=args.atomic_db,
            bundle_db=args.bundle_db,
            freq_channel=args.freq_channel,
            wafer=args.wafer,
            bundle_id=bundle_id,
            null_prop_val=split_inter_obs,
            pix_type=args.pix_type,
            car_map_template=args.car_map_template,
            split_label=split_intra_obs,
            map_dir=args.map_dir
        )
        
        print(len(signflipper.fnames))

        # NEW check here
        if not signflipper.fnames:
            print(f"Skipping bundle_id={bundle_id} sim_id={sim_id} (no atomic files)")
            continue

        # Make noise map
        noise_map, noise_weight = signflipper.signflip(seed=12345*bundle_id + sim_id)

        # Output filenames
        out_fname = os.path.join(
            out_dir,
            args.map_string_format.format(
                split=split_tag,
                bundle_id=bundle_id,
                wafer=wafer_tag,
                patch=patch_tag,
                freq_channel=args.freq_channel
            )
        )
        out_fname = out_fname.replace("__", "_")
        out_fname = out_fname.replace("_map.fits", f"_{sim_id:04d}_map.fits")
        os.makedirs(os.path.dirname(out_fname), exist_ok=True)

        # Save maps
        if args.pix_type == "car":
            enmap.write_map(out_fname, noise_map)
            enmap.write_map(out_fname.replace("map.fits", "weights.fits"), noise_weight)
            print(out_fname)

            # Quickplots
            if sim_id % (n_sims // 3) == 0:
                plot = enplot.plot(noise_map*1e6, colorbar=True, min=-50, max=50, ticks=10)
                for ip, p in enumerate(["Q", "U"]):
                    enplot.write(out_fname.replace(".fits", f"{p}.png"), plot[ip+1])

        else:  # HEALPix
            hp.write_map(out_fname, noise_map, overwrite=True, dtype=np.float32)
            hp.write_map(out_fname.replace("map.fits", "weights.fits"), noise_weight, overwrite=True, dtype=np.float32)

            if sim_id % (n_sims // 3) == 0:
                for ip, p in enumerate(["Q", "U"]):
                    hp.mollview(noise_map[ip+1]*1e6, cmap="RdYlBu_r",
                                title=f"{p} Noise {bundle_id} sim{sim_id}",
                                min=-100, max=100, unit=r"$\mu$K")
                    plt.savefig(out_fname.replace(".fits", f"{p}.png"))
                    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make bundled noise maps.")
    parser.add_argument("--config_file", type=str, help="yaml file with configuration.")
    args = parser.parse_args()

    config = bundling_utils.Cfg.from_yaml(args.config_file)
    its = [np.atleast_1d(x) for x in [config.freq_channel, config.wafer]]
    patch_list = config.patch

    for patch in np.atleast_1d(patch_list):
        patch_tag = "" if patch is None else patch
        bundle_db = config.bundle_db.format(patch=patch_tag, seed=config.seed).replace("__", "_")

        if config.only_make_db:
            config1 = config.copy()
            config1.patch = patch
            config1.bundle_db = bundle_db
            main(config1)
        else:
            for it in itertools.product(*its):
                config1 = config.copy()
                config1.patch = patch
                config1.bundle_db = bundle_db
                config1.freq_channel, config1.wafer = it

                # Inter-obs
                if config1.inter_obs_splits is not None:
                    config2 = config1.copy()
                    config2.split_label_intra_obs = config2.intra_obs_pair
                    for null_prop_val in config2.inter_obs_splits:
                        config2.null_prop_val_inter_obs = null_prop_val
                        try:
                            main(config2)
                        except ValueError as e:
                            print(e)

                # Intra-obs
                if config1.intra_obs_splits is not None:
                    config2 = config1.copy()
                    config2.null_prop_val_inter_obs = None
                    for split_val in config2.intra_obs_splits:
                        config2.split_label_intra_obs = split_val
                        try:
                            main(config2)
                        except ValueError as e:
                            print(e)
