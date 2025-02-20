import argparse
from bundling_utils import Cfg
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
    patch = args.patch
    query_restrict=" ".join(args.query_restrict)
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
            seed=args.seed, null_props=args.inter_obs,
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
        atomic_list = np.load(args.atomic_list)["atomic_list"]

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
            name_tag = f"{args.freq_channel}_science"
        elif (split_intra_obs is not None) and (split_inter_obs is not None):
            # Assume this is an inter with summed intras
            name_tag = f"{args.freq_channel}_{split_inter_obs}"
        elif split_intra_obs is not None:
            if isinstance(split_intra_obs, list):
                name_tag = f"{args.freq_channel}_{'_'.join(split_intra_obs)}"
            else:
                name_tag = f"{args.freq_channel}_{split_intra_obs}"
        elif split_inter_obs is not None:
            name_tag = f"{args.freq_channel}_{split_inter_obs}"

        out_fname = os.path.join(
            out_dir,
            args.map_string_format.format(name_tag=name_tag,
                                          bundle_id=bundle_id)
        )

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
    config = Cfg.from_yaml(args.config_file)

    its = [np.atleast_1d(x) for x in [config.patch, config.freq_channel, config.wafer]]
    for it in itertools.product(*its):
        patch, freq_channel, wafer = it
        config.patch = patch
        config.freq_channel = freq_channel
        config.wafer = wafer

        # Inter-obs
        config.split_label_intra_obs = config.split_labels_inter
        for null_prop_val in config.null_prop_vals:
            config.null_prop_val_inter_obs = null_prop_val
            main(config)

        # Intra-obs
        for split_val in config.split_labels:
            config.split_label_intra_obs = split_val
            main(config)
