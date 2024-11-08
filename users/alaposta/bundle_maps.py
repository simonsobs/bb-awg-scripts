import argparse
from coadder import Bundler
import os
import healpy as hp
from coordinator import BundleCoordinator

def main(args):
    """
    """
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    bundler = Bundler(args.atomic_db, args.bundle_db, args.freq_channel, args.wafer)
    bundle_coordinator = BundleCoordinator.from_dbfile(args.bundle_db)
    bundle_ids = list(set(bundle_coordinator.bundle_id))

    for bundle_id in bundle_ids:
        bundled_map, hits_map = bundler.bundle(bundle_id, null_prop_val=args.null_prop_val)
        out_fname = os.path.join(out_dir, f"satp1_{args.freq_channel}_bundle{bundle_id}_map.fits")
        hp.write_map(out_fname, bundled_map, overwrite=True)
        hp.write_map(out_fname.replace("map.fits", "hits.fits"), hits_map, overwrite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create bundle maps")
    parser.add_argument("--bundle-db", help="Bundle database")
    parser.add_argument("--atomic-db", help="Atomic database")
    parser.add_argument("--freq-channel", help="Frequency channel")
    parser.add_argument("--wafer", help="Wafer", default=None)
    parser.add_argument("--null-prop-val", help="Null property value", default=None)
    parser.add_argument("--output-dir", help="Output directory")

    args = parser.parse_args()

    main(args)