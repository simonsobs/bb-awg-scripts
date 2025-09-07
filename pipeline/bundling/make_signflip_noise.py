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

sys.path.append("/home/ccaimapo/SimonsObs/bb-awg-scripts/pipeline/misc")
import mpi_utils as mpi  # noqa

def main(args, size, rank, comm):
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
    sat_bundle_dbs = np.atleast_1d(getattr(args, "bundle_db", []))
    sat_atomic_dbs = np.atleast_1d(getattr(args, "atomic_db", []))
    sat_map_dirs = np.atleast_1d(args.map_dir)

    # Validate each bundle_db exists before proceeding
    for db in sat_bundle_dbs:
        if not os.path.isfile(db):
            raise FileNotFoundError(f"File {db} does not exist.")

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
    if rank==0:
        print(f"{n_missing} tasks missing out of {args.n_bundles * n_sims} total.")
    if n_missing == 0:
        return

    task_ids = mpi.distribute_tasks(size, rank, n_missing,)
    #local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    # Loop over only missing tasks
    #for task_id in mpi.taskrange(len(missing_tasks)):
    #for task_id in mpi.taskrange(n_missing - 1):
    for task_id in task_ids:
        bundle_id, sim_id = missing_tasks[task_id]
        print(f"Running bundle_id={bundle_id} sim_id={sim_id}")

        combined_map = None
        combined_weight = None
        
        # Now construct the signflipper (only for needed bundle_id)
        for atomic_db, bundle_db, map_dir_i in zip(sat_atomic_dbs, sat_bundle_dbs, sat_map_dirs):
            signflipper = SignFlipper(
                atomic_db=atomic_db,
                bundle_db=bundle_db,
                freq_channel=args.freq_channel,
                wafer=args.wafer,
                bundle_id=bundle_id,
                null_prop_val=split_inter_obs,
                pix_type=args.pix_type,
                car_map_template=args.car_map_template,
                split_label=split_intra_obs,
                map_dir=map_dir_i
            )
        
            print(len(signflipper.fnames))

            # NEW check here
            if not signflipper.fnames:
                print(f"Skipping bundle_id={bundle_id} sim_id={sim_id} (no atomic files)")
                continue

            # Make noise map
            try:
                result = signflipper.signflip(seed=12345 * bundle_id + sim_id)
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    print(f"SignFlipper returned unexpected result for bundle_id={bundle_id} sim_id={sim_id} from {bundle_db}: {result}")
                    continue
                noise_map, noise_weight = result
            except Exception as e:
                print(f"SignFlipper crashed for bundle_id={bundle_id} sim_id={sim_id} from {bundle_db}: {e}")
                continue
            
            # Initialize or accumulate
            if combined_map is None:
                combined_map = noise_map.copy()
                combined_weight = noise_weight.copy()
            else:
                combined_map += noise_map
                combined_weight += noise_weight
                
            if combined_map is None:
                print(f"No valid maps for bundle_id={bundle_id} sim_id={sim_id}, skipping save.")
                exut()
 

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
            enmap.write_map(out_fname, combined_map)
            enmap.write_map(out_fname.replace("map.fits", "weights.fits"), combined_weight)
            print(out_fname)

            # Quickplots
            if sim_id % (n_sims // 3) == 0:
                plot = enplot.plot(combined_map*1e6, colorbar=True, min=-50, max=50, ticks=10)
                for ip, p in enumerate(["Q", "U"]):
                    enplot.write(out_fname.replace(".fits", f"{p}.png"), plot[ip+1])

        else:  # HEALPix
            hp.write_map(out_fname, combined_map, overwrite=True, dtype=np.float32)
            hp.write_map(out_fname.replace("map.fits", "weights.fits"), combined_weight, overwrite=True, dtype=np.float32)

            if sim_id % (n_sims // 3) == 0:
                for ip, p in enumerate(["Q", "U"]):
                    hp.mollview(combined_map[ip+1]*1e6, cmap="RdYlBu_r",
                                title=f"{p} Noise {bundle_id} sim{sim_id}",
                                min=-100, max=100, unit=r"$\mu$K")
                    plt.savefig(out_fname.replace(".fits", f"{p}.png"))
                    plt.close()
    comm.Barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make bundled noise maps.")
    parser.add_argument("--config_file", type=str, help="YAML file with configuration.")
    args = parser.parse_args()

    # Load config
    config = bundling_utils.Cfg.from_yaml(args.config_file)

    # Build frequency x wafer combinations
    its = [np.atleast_1d(x) for x in [config.freq_channel, config.wafer]]
    patch_list = config.patch

    # MPI initialization
    rank, size, comm = mpi.init(True)

    for patch in np.atleast_1d(patch_list):
        patch_tag = "" if patch is None else patch

        if config.only_make_db:
            config1 = config.copy()
            config1.patch = patch
            main(config1, size, rank, comm)
        else:
            for it in itertools.product(*its):
                config1 = config.copy()
                config1.patch = patch
                config1.freq_channel, config1.wafer = it

                # Inter-obs splits
                if config1.inter_obs_splits is not None:
                    config2 = config1.copy()
                    config2.split_label_intra_obs = config2.intra_obs_pair
                    for null_prop_val in config2.inter_obs_splits:
                        config2.null_prop_val_inter_obs = null_prop_val
                        try:
                            main(config2, size, rank, comm)
                        except ValueError as e:
                            print(e)

                # Intra-obs splits
                if config1.intra_obs_splits is not None:
                    config2 = config1.copy()
                    config2.null_prop_val_inter_obs = None
                    for split_val in config2.intra_obs_splits:
                        config2.split_label_intra_obs = split_val
                        try:
                            main(config2, size, rank, comm)
                        except ValueError as e:
                            print(e)
