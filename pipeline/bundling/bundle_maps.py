import argparse
from coadder import Bundler
import os
import healpy as hp
from pixell import enmap, enplot, reproject
import numpy as np
from coordinator import BundleCoordinator

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
    if args.pix_type not in ["hp", "car"]:
        raise ValueError(
            "Unknown pixel type, must be 'car' or 'hp'."
        )

    if args.null_prop_val_inter_obs in ["None", "none", "science"]:
        args.null_prop_val_inter_obs = None

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    car_map_template = args.car_map_template

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
            seed=args.seed, null_props=args.null_props
        )
        bundle_coordinator.save_db(args.bundle_db)

    if args.only_make_db:
        return

    atomic_list = None
    if args.atomic_list is not None:
        atomic_list = np.load(args.atomic_list)["atomic_list"]

    bundler = Bundler(
        atomic_db=args.atomic_db,
        bundle_db=args.bundle_db,
        freq_channel=args.freq_channel,
        wafer=args.wafer,
        pix_type=args.pix_type,
        atomic_list=atomic_list,
        car_map_template=car_map_template
    )

    # DEBUG
    print("len(bundler.atomic_list) =", len(bundler.atomic_list))

    bundle_ids = range(args.n_bundles)

    for bundle_id in bundle_ids:
        print(" - bundle_id", bundle_id)
        bundled_map, hits_map = bundler.bundle(
            bundle_id,
            split_label=args.split_label_intra_obs,
            null_prop_val=args.null_prop_val_inter_obs
        )
        if args.null_prop_val_inter_obs is not None:
            name_tag = f"{args.freq_channel}_{args.null_prop_val_inter_obs}"
        elif args.split_label_intra_obs is not None:
            name_tag = f"{args.freq_channel}_{args.split_label_intra_obs}"
        elif (args.null_prop_val_inter_obs is not None and
              args.split_label_intra_obs is not None):
            raise ValueError(
                "Both split types cannot be selected at the same time."
            )
        else:
            name_tag = f"{args.freq_channel}_science"
        out_fname = os.path.join(
            out_dir,
            args.map_string_format.format(name_tag=name_tag,
                                          bundle_id=bundle_id)
        )

        if args.pix_type == "car":
            enmap.write_map(out_fname, bundled_map)
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
                    min=-1, max=1, ticks=10
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
                val = args.null_prop_val
                label = "" if val is None else f", {val}"
                enplot.write(
                    out_fname.replace(".fits", f"{p}.png"),
                    plot[ip+1]
                )

        elif args.pix_type == "hp":
            hp.write_map(
                out_fname, bundled_map, overwrite=True, dtype=np.float32
            )
            hp.write_map(
                out_fname.replace("map.fits", "_norm_hits.fits"), hits_map,
                overwrite=True, dtype=np.float32
            )
            for ip, p in enumerate(["Q", "U"]):
                val = args.null_prop_val
                label = "" if val is None else f", {val}"
                hp.mollview(
                    bundled_map[ip+1]*1e6, cmap="RdYlBu_r",
                    title=f"{p} Bundle {bundle_id}{label}",
                    min=-100, max=100, unit=r"$\mu$K"
                )
                plt.savefig(out_fname.replace(".fits", f"{p}.png"))
                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make bundled maps")

    parser.add_argument(
        "--bundle_db",
        help="Bundle database."
    )
    parser.add_argument(
        "--atomic_db",
        help="Atomic map database."
    )
    parser.add_argument(
        "--atomic_list",
        help="Optional list of atomic maps to further restrict the atomic_db.",
        default=None
    )
    parser.add_argument(
        "--freq_channel",
        help="Frequency channel, e.g. 'f090'"
    )
    parser.add_argument(
        "--wafer",
        help="Wafer label, e.g. 'ws0'.",
        default=None
    )
    parser.add_argument(
        "--n_bundles",
        help="Number of map bundles.",
        type=int,
        required=True
    )
    parser.add_argument(
        "--null_prop_val_inter_obs",
        help="Null property value for inter-obs splits, e.g. 'pwv_low'.",
        default=None
    )
    parser.add_argument(
        "--split_label_intra_obs",
        help="Split label for intra-obs splits, e.g. 'scans_left'.",
        default=None
    )
    parser.add_argument(
        "--pix_type",
        help="Pixel type, either 'hp' or 'car'.",
        default="hp"
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory."
    )
    parser.add_argument(
        "--map_string_format",
        help="String formatting; must contain {name_tag} and {bundle_id}."
    )
    parser.add_argument(
        "--car_map_template",
        help="Path to CAR coadded (hits) map to be used as geometry template.",
        default=None
    )
    parser.add_argument(
        "--seed",
        help="Random seed that determines the composition of bundles.",
        type=int, default=1234
    )
    parser.add_argument(
        "--null_props",
        nargs="*",
        default=None,
        help="Null properties for bundling database, e.g. 'pwv elevation'."
    )
    parser.add_argument(
        "--only_make_db",
        action="store_true",
        help="Only make bundling database and do not bundle maps?"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite database if exists?"
    )

    args = parser.parse_args()

    main(args)
