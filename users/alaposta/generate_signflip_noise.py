import argparse
from coadder import SignFlip
import os
import healpy as hp
from coordinator import BundleCoordinator

def main(args):
    """
    """
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    bundle_coordinator = BundleCoordinator.from_dbfile(args.bundle_db)
    bundle_ids = list(set(bundle_coordinator.bundle_id))

    signflips = {}

    for bundle_id in bundle_ids:
        signflip = SignFlip(
            args.atomic_db,
            args.bundle_db,
            freq_channel=args.freq_channel,
            wafer=args.wafer,
            bundle_id=bundle_id,
            null_prop_val=args.null_prop_val)
        signflips[bundle_id] = signflip

    idmin, idmax = args.sim_ids.split(",")
    idmin, idmax = int(idmin), int(idmax)

    # TODO: parallelize + optimize sum process in coadder.py
    for sim_id in range(idmin, idmax):
        for bundle_id in bundle_ids:
            noise = signflips[bundle_id].signflip(seed=sim_id)
            if args.null_prop_val is not None:
                name_tag = f"{args.freq_channel}_{args.null_prop_val}"
            else:
                name_tag = f"{args.freq_channel}"
            fname = f"satp1_{name_tag}_bundle{bundle_id}_noise_map_{sim_id:04d}.fits"
            out_fname = os.path.join(out_dir, fname)
            hp.write_map(out_fname, noise, overwrite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create bundle maps")
    parser.add_argument("--bundle-db", help="Bundle database")
    parser.add_argument("--atomic-db", help="Atomic database")
    parser.add_argument("--freq-channel", help="Frequency channel")
    parser.add_argument("--wafer", help="Wafer", default=None)
    parser.add_argument("--null-prop-val", help="Null property value", default=None)
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--sim-ids", help="Comma separated simulation IDs", type=str)

    args = parser.parse_args()

    main(args)