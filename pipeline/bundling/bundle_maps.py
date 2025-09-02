import argparse
import bundling_utils
from coadder import Bundler
import os
import healpy as hp
from pixell import enmap, enplot, reproject
import numpy as np
from coordinator import BundleCoordinator
import itertools

import matplotlib.pyplot as plt


def car2healpix(norm_hits_map):
    """
    Tranforms an intensive spin-0 map (e.g. hits map normalized to maximum 1)
    into a healpix map.
    """
    return reproject.map2healpix(norm_hits_map, spin=[0])


def main(args):
    """
    """
    args = args.copy()  # Make sure we don't accidentally modify the input args
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

    if os.path.isfile(args.bundle_db) and not args.overwrite:
        print(f"Loading from {args.bundle_db}.")
        bundle_coordinator = BundleCoordinator.from_dbfile(
            args.bundle_db,
            null_prop_val=args.null_prop_val_inter_obs
        )
    else:
        print(f"Writing to {args.bundle_db}.")
        bundle_coordinator = BundleCoordinator(
            args.atomic_db, n_bundles=args.n_bundles,
            seed=args.seed, null_props=args.inter_obs_props,
            query_restrict=query_restrict
        )
        bundle_coordinator.save_db(args.bundle_db)

    if args.only_make_db:
        return

    if args.pix_type not in ["hp", "car"]:
        raise ValueError(
            "Unknown pixel type, must be 'car' or 'hp'."
        )

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    atomic_list = None
    if args.atomic_list is not None:
        atomic_list = np.load(args.atomic_list)

    car_map_template = args.car_map_template

    bundler = Bundler(
        atomic_db=args.atomic_db,
        bundle_db=args.bundle_db,
        freq_channel=args.freq_channel,
        wafer=args.wafer,
        pix_type=args.pix_type,
        atomic_list=atomic_list,
        car_map_template=car_map_template,
        telescope=args.tel,
        query_restrict=query_restrict
    )

    bundle_ids = range(args.n_bundles)

    for bundle_id in bundle_ids:
        print(" - bundle_id", bundle_id)
        split_intra_obs, split_inter_obs = (args.split_label_intra_obs,
                                            args.null_prop_val_inter_obs)

        bundled_map, weights_map, hits_map, fnames = bundler.bundle(
            bundle_id,
            split_label=split_intra_obs,
            null_prop_val=split_inter_obs,
            map_dir=args.map_dir,
            abscal=args.abscal,
            nproc=args.nproc
        )

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
        # For plot titles
        name_tag = f"{args.freq_channel}_{wafer_tag}_{patch_tag}_{split_tag}"
        name_tag = name_tag.replace("__", "_")

        os.makedirs(os.path.dirname(out_fname), exist_ok=True)
        if args.save_fnames:
            out_filenames = out_fname.replace("map.fits", "fnames.txt")
            np.savetxt(out_filenames, fnames, fmt='%s')
        if args.pix_type == "car":
            enmap.write_map(out_fname, bundled_map)
            enmap.write_map(out_fname.replace("map.fits", "weights.fits"),
                            weights_map)
            enmap.write_map(out_fname.replace("map.fits", "hits.fits"),
                            hits_map)
            plot = enplot.plot(
                bundled_map*1e6, colorbar=True,
                min=-100, max=100, ticks=10
            )
            enplot.write(
                out_fname.replace(".fits", "_hits"),
                enplot.plot(
                    hits_map, colorbar=True,
                    ticks=10
                )
            )
            enplot.write(
                out_fname.replace(".fits", "_norm_hits"),
                enplot.plot(
                    hits_map/np.max(hits_map, axis=(0, 1)), colorbar=True,
                    min=-1, max=1, ticks=10
                )
            )
            for ip, p in enumerate(["Q", "U"]):
                enplot.write(
                    out_fname.replace(".fits", f"{p}.png"),
                    plot[ip+1]
                )

        elif args.pix_type == "hp":
            hp.write_map(
                out_fname, bundled_map, overwrite=True, dtype=np.float64
            )
            hp.write_map(
                out_fname.replace("map.fits", "weights.fits"), weights_map,
                overwrite=True, dtype=np.float64
            )
            hp.write_map(
                out_fname.replace("map.fits", "hits.fits"), hits_map,
                overwrite=True, dtype=np.float64
            )
            for ip, p in enumerate(["Q", "U"]):
                hp.mollview(
                    bundled_map[ip+1]*1e6, cmap="RdYlBu_r",
                    title=f"{p} Bundle {bundle_id} {name_tag}",
                    min=-100, max=100, unit=r"$\mu$K"
                )
                plt.savefig(out_fname.replace(".fits", f"{p}.png"))
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

                # Make full coadds
                if config1.coadd_split_pair is not None:
                    print("Making full maps")

                    savename = template.format(config1.coadd_splits_name, "{}", "{}")
                    bundling_utils.make_full(template,
                                             config1.coadd_split_pair,
                                             config1.n_bundles,
                                             config1.pix_type,
                                             do_hits=True,
                                             savename=savename,
                                             return_maps=False)

                if config1.coadd_bundles_splitname is not None:
                    print("Co-adding bundles")
                    for coadd_bundles_splitname in np.atleast_1d(config1.coadd_bundles_splitname):
                        print(coadd_bundles_splitname)
                        temp = template.format(coadd_bundles_splitname, "{}", "{}")
                        sum_vals = list(range(config1.n_bundles))
                        savename = temp.format("!", "{}").replace("_bundle!", "")
                        coadd_map, _, _ = bundling_utils.coadd_bundles(temp,
                                                                       sum_vals,
                                                                       config1.pix_type,
                                                                       do_hits=True,
                                                                       savename=savename)

                        plot = enplot.plot(coadd_map*1e6, colorbar=True, color='gray', range="100:20:20", ticks=10, downgrade=2, autocrop=True)
                        enplot.write(savename.replace("{}.fits", "mapQ.png"), plot[1])
                        enplot.write(savename.replace("{}.fits", "mapU.png"), plot[2])
