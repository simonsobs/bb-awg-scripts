import argparse
import os
import sys
import numpy as np
import itertools
from coadder import SignFlipper
import bundling_utils as utils

bundling_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(bundling_dir, "../misc"))

import mpi_utils as mpi  # noqa

def _make_signflip(args, size, rank, comm, split_intra_obs=None, split_inter_obs=None):
    """
    Main function to create sign-flipped noise realizations.
    """
    args = args.copy()  # Make sure we don't modify input

    # Read bundle.db
    sat_bundle_dbs = np.atleast_1d(getattr(args, "bundle_db_full", []))
    sat_map_dirs = np.atleast_1d(args.map_dir)

    # Validate each bundle_db exists before proceeding
    for db in sat_bundle_dbs:
        if not os.path.isfile(db):
            raise FileNotFoundError(f"File {db} does not exist.")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    bundle_ids = range(args.n_bundles)
    #n_sims = config.n_sims
    n_sims = getattr(args, "n_sims", None) or getattr(globals().get("config", object()), "n_sims", None)
    if n_sims is None:
        raise ValueError("n_sims not found on args or global config.")

    split_tag = utils.get_split_tag(split_intra_obs, split_inter_obs, args.intra_obs_pair, args.coadd_splits_name)
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
                    freq_channel=args.freq_channel,
                    map_type="{map_type}"
                )
            )
            out_fname = out_fname.replace("{map_type}", f"{sim_id:04d}"+"_{map_type}")
            out_fname = out_fname.replace("__", "_")
            if not os.path.exists(out_fname.format(map_type="map")) or args.overwrite:
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
        for bundle_db, map_dir_i in zip(sat_bundle_dbs, sat_map_dirs):
            signflipper = SignFlipper(
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
                continue

        # Output filenames
        out_fname = os.path.join(
            out_dir,
            args.map_string_format.format(
                split=split_tag,
                bundle_id=bundle_id,
                wafer=wafer_tag,
                patch=patch_tag,
                freq_channel=args.freq_channel,
                map_type='{map_type}'
            )
        )
        out_fname = out_fname.replace("__", "_")
        out_fname = out_fname.replace("{map_type}", f"{sim_id:04d}"+"_{}")

        # Skip existing maps if overwrite=False
        if (not args.overwrite) and os.path.exists(out_fname.format("map")):
            if rank == 0:
                print(f"Skipping existing: {out_fname}")
            continue

        # Save maps
        print(out_fname)
        utils.write_maps(out_fname, args.pix_type, combined_map, combined_weight, dtype=np.float32)

        # Quickplots
        if sim_id % (n_sims // 3) == 0:
            savename_plot = out_fname[:out_fname.find(".fits")] + ".png"
            utils.plot_map(savename_plot.format("Q"), args.pix_type, combined_map[1], unit_fac=1e6, vrange=50)
            utils.plot_map(savename_plot.format("U"), args.pix_type, combined_map[2], unit_fac=1e6, vrange=50)

    comm.Barrier()

def make_signflip(args, size, rank, comm, split_intra_obs=None, split_inter_obs=None):
    try:
        _make_signflip(args, size, rank, comm, split_intra_obs, split_inter_obs)
    except ValueError as e:
        print(e)


def main(args):
    # Load config
    config = utils.Cfg.from_yaml(args.config_file)

    # Build frequency x wafer combinations
    its = [np.atleast_1d(x) for x in [config.freq_channel, config.wafer]]
    patch_list = config.patch_list

    # MPI initialization
    rank, size, comm = mpi.init(True)

    for patch in np.atleast_1d(patch_list):
        if config.only_make_db:
            config_db = utils.child_config(config, patch=patch)
            make_signflip(config_db, size, rank, comm)
        else:
            for it in itertools.product(*its):
                config_it = utils.child_config(config, patch=patch, freq_channel=it[0], wafer=it[1])
                intra_pair = config_it.intra_obs_pair

                # --- science/full run: coadd the pair, no inter-obs null ---
                make_signflip(config_it, size, rank, comm, split_intra_obs=intra_pair, split_inter_obs=None)

                # Inter-obs splits
                if config_it.inter_obs_splits is not None:
                    for null_prop_val in config_it.inter_obs_splits:
                        make_signflip(config_it, size, rank, comm, split_intra_obs=intra_pair, split_inter_obs=null_prop_val)

                # Intra-obs splits
                if config_it.intra_obs_splits is not None:
                    for split_val in config_it.intra_obs_splits:
                        make_signflip(config_it, size, rank, comm, split_intra_obs=split_val, split_inter_obs=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make bundled noise maps.")
    parser.add_argument("--config_file", type=str, help="YAML file with configuration.")
    args = parser.parse_args()

    main(args)
