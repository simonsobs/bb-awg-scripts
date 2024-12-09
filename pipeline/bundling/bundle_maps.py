import argparse
from coadder import Bundler
import os
import healpy as hp
import numpy as np
from coordinator import BundleCoordinator

import matplotlib.pyplot as plt


def main(args):
    """
    """
    if args.pix_type != "hp":
        raise NotImplementedError("Only accepting hp as input for now.")

    if args.null_prop_val == "None":
        args.null_prop_val = None

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(args.bundle_db):
        print(f"Loading from {args.bundle_db}.")
        bundle_coordinator = BundleCoordinator.from_dbfile(
            args.bundle_db, null_prop_val=args.null_prop_val
        )
    else:
        print(f"Writing to {args.bundle_db}.")
        bundle_coordinator = BundleCoordinator(
            args.atomic_db, n_bundles=args.n_bundles,
            seed=1234, null_props=["pwv", "elevation"]
        )
        bundle_coordinator.save_db(args.bundle_db)
        print("Median PWV:", bundle_coordinator.null_props_stats["pwv"])

    bundler = Bundler(
        atomic_db=args.atomic_db,
        bundle_db=args.bundle_db,
        freq_channel=args.freq_channel,
        wafer=args.wafer,
        pix_type=args.pix_type
    )

    bundle_ids = range(args.n_bundles)

    for bundle_id in bundle_ids:
        print(" - bundle_id", bundle_id)
        bundled_map, hits_map = bundler.bundle(
            bundle_id, null_prop_val=args.null_prop_val
        )
        if args.null_prop_val is not None:
            name_tag = f"{args.freq_channel}_{args.null_prop_val}"
        else:
            name_tag = f"{args.freq_channel}"
        out_fname = os.path.join(
            out_dir,
            args.map_string_format.format(name_tag=name_tag,
                                          bundle_id=bundle_id)
        )
        # FIXME: Generalize to CAR input

        # HEALPix input
        hp.write_map(out_fname, bundled_map, overwrite=True, dtype=np.float32)
        hp.write_map(out_fname.replace("map.fits", "hits.fits"), hits_map,
                     overwrite=True, dtype=np.float32)
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
    parser.add_argument("--bundle_db", help="Bundle database")
    parser.add_argument("--atomic_db", help="Atomic database")
    parser.add_argument("--freq_channel", help="Frequency channel")
    parser.add_argument("--wafer", help="Wafer", default=None)
    parser.add_argument("--n_bundles", help="Number of bundles", type=int,
                        required=True)
    parser.add_argument("--null_prop_val", help="Null property value",
                        default=None)
    parser.add_argument("--pix_type", help="Pixel type, either hp or car",
                        default="hp")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--map_string_format",
                        help="String formatting. Must contain {name_tag}"
                             " and {bundle_id}.")

    args = parser.parse_args()

    main(args)
