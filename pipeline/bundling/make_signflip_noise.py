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
    """
    args = args_dict = args.copy()  # Convert Namespace to dict and copy it  # Make sure we don't accidentally modify the input args
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
            raise ValueError(f"patch {patch} not recognized.")
    
    # Read bundle.db
    if os.path.isfile(args.bundle_db) and not args.overwrite:
        print(f"Loading from {args.bundle_db}.")
        bundle_coordinator = BundleCoordinator.from_dbfile(
            args.bundle_db,
            null_prop_val=args.null_prop_val_inter_obs
        )
    else:
        raise FileNotFoundError(f"File {args.bundle_db} does not exist.")
        # TODO: we don't need a db here right? 
        #print(f"Writing to {args.bundle_db}.")
        #bundle_coordinator = BundleCoordinator(
        #    args.atomic_db, n_bundles=args.n_bundles,
        #    seed=args.seed, null_props=args.inter_obs_props,
        #    query_restrict=query_restrict
        #)
        #bundle_coordinator.save_db(args.bundle_db)
    
    if args.pix_type not in ["hp", "car"]:
        raise ValueError(
            "Unknown pixel type, must be 'car' or 'hp'."
        )

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    atomic_list = None
    if args.atomic_list is not None:
        atomic_list = np.load(args.atomic_list)["atomic_list"]

    car_map_template = args.car_map_template

    bundle_ids = range(args.n_bundles)

    mpi.init(True)

    for bundle_id in bundle_ids:
        print(" - bundle_id", bundle_id)
        split_intra_obs, split_inter_obs = (args.split_label_intra_obs,
                                            args.null_prop_val_inter_obs)
        
        print(split_intra_obs, split_inter_obs)
        
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
        
        for sim_id in mpi.taskrange(config.n_sims - 1):
            print("    sim_id", sim_id)
            noise_map, noise_weight = signflipper.signflip(seed=12345*bundle_id+sim_id)

            # Map naming convention
            if (split_intra_obs, split_inter_obs) == (None, None):
                split_tag = "science"
            elif split_inter_obs is not None:
                # Inter; potentially with summed intras
                split_tag = split_inter_obs
            elif split_intra_obs is not None:
                split_tag = split_intra_obs
            if isinstance(split_tag, list):
                split_tag = '_'.join(split_tag)

            wafer_tag = args.wafer if args.wafer is not None else ""
            patch_tag = args.patch if args.patch is not None else ""

            for required_tag in ["{split}", "{bundle_id}", "{freq_channel}"]:
                if required_tag not in args.map_string_format:
                    raise ValueError(f"map_string_format does not have \
                                    required placeholder {required_tag}")
            for optional_tag, tag_val in zip(["{wafer}", "{patch}"], [wafer_tag, patch_tag]):
                if optional_tag not in args.map_string_format and tag_val:
                    print(f"Warning: map_string_format does not have optional \
                        placeholder {optional_tag} but value is passed")

            out_fname = os.path.join(
                out_dir,
                args.map_string_format.format(split=split_tag,
                                            bundle_id=bundle_id,
                                            wafer=wafer_tag,
                                            patch=patch_tag,
                                            freq_channel=args.freq_channel)
            )
            
            # Again hacky removal of hopefully accidental double underscores
            out_fname = out_fname.replace("__", "_")
            out_fname = out_fname.replace("_map.fits", f"_{sim_id:04d}_map.fits")
            # For plot titles
            name_tag = f"{args.freq_channel}_{wafer_tag}_{patch_tag}_{split_tag}"
            name_tag = name_tag.replace("__", "_")

            os.makedirs(os.path.dirname(out_fname), exist_ok=True)
            
            if args.pix_type == "car":
                enmap.write_map(out_fname, noise_map)
                enmap.write_map(out_fname.replace("map.fits", "weights.fits"),
                                noise_weight)
                
                # Plot a couple of noise realizations
                if sim_id % (args.n_sims // 3) != 0:
                    continue
                
                plot = enplot.plot(
                    noise_map*1e6, colorbar=True,
                    min=-50, max=50, ticks=10
                )
                for ip, p in enumerate(["Q", "U"]):
                    enplot.write(
                        out_fname.replace(".fits", f"{p}.png"),
                        plot[ip+1]
                    )
            else:
                # HEALPix input
                hp.write_map(out_fname, noise_map, overwrite=True,
                            dtype=np.float32)
                hp.write_map(out_fname.replace("map.fits", "weights.fits"),
                                noise_weight, overwrite=True, dtype=np.float32)

                # Plot a couple of noise realizations
                if sim_id % (args.n_sims // 3) != 0:
                    continue

                os.makedirs(f"{out_dir}/plots", exist_ok=True)
                plot_fname = out_fname.replace(".fits", f"{p}.png")

                for ip, p in enumerate(["Q", "U"]):
                    hp.mollview(
                        noise_map[ip+1]*1e6, cmap="RdYlBu_r",
                        title=f"{p} Noise {bundle_id}{label} sim{sim_id}",
                        min=-100, max=100, unit=r"$\mu$K"
                    )
                    plt.savefig(plot_fname.replace(".fits", f"{p}.png"))
                    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make bundled maps")
    parser.add_argument(
        "--config_file", type=str, help="yaml file with configuration."
    )

    args = parser.parse_args()
    config = bundling_utils.Cfg.from_yaml(args.config_file)
    its = [np.atleast_1d(x) for x in [config.freq_channel, config.wafer]]
    patch_list = config.patch

    for patch in np.atleast_1d(patch_list):
        patch_tag = "" if patch is None else patch
        bundle_db = config.bundle_db.format(patch=patch_tag,
                                            seed=config.seed)
        # Hacky but remove any (presumed accidental) double underscores
        bundle_db = bundle_db.replace("__", "_")

        # Make db only
        if config.only_make_db:
            config1 = config.copy()
            config1.patch = patch
            config1.bundle_db = bundle_db
            main(config1)

        else:
            for it in itertools.product(*its):
                print(patch, it)

                config1 = config.copy()
                config1.patch = patch
                config1.bundle_db = bundle_db
                config1.freq_channel, config1.wafer = it

                # Inter-obs
                if config1.inter_obs_splits is not None:
                    config2 = config1.copy()
                    config2.split_label_intra_obs = config2.intra_obs_pair
                    for null_prop_val in config2.inter_obs_splits:
                        print(null_prop_val)
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
                        print(split_val)
                        config2.split_label_intra_obs = split_val
                        try:
                            main(config2)
                        except ValueError as e:
                            print(e)

                wafer_tag = "" if config1.wafer is None else config1.wafer
                template = os.path.join(config1.output_dir,
                                        config1.map_string_format.format(split="{}",
                                                                         bundle_id="{}",
                                                                         wafer=wafer_tag,
                                                                         patch=patch_tag,
                                                                         freq_channel=config1.freq_channel
                                                                         ))
                template = template.replace("__", "_")
                template = template.replace("map.fits", "{}.fits")
