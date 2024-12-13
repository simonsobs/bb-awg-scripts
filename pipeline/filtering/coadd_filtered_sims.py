import numpy as np
import healpy as hp
import argparse
import sqlite3
import os
import sys
import matplotlib.pyplot as plt

sys.path.append("../bundling")
sys.path.append("../misc")
from coordinator import BundleCoordinator  # noqa


def main(args):
    """
    """
    # ArgumentParser
    out_dir = args.output_directory
    os.makedirs(out_dir, exist_ok=True)

    plot_dir = f"{out_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    atomics_dir = f"{out_dir}/atomics_sims"
    os.makedirs(atomics_dir, exist_ok=True)

    # Databases
    atom_db = args.atomic_db
    bundle_db = args.bundle_db

    # Sim related arguments
    nside = args.nside  # FIXME: generalize to CAR adn/or HEALPix
    map_template = args.map_template
    sim_ids = args.sim_ids

    if "," in sim_ids:
        id_min, id_max = sim_ids.split(",")
    else:
        id_min = sim_ids
        id_max = id_min
    id_min, id_max = (int(id_min), int(id_max))

    # Bundle query arguments
    freq_channel = args.freq_channel
    null_prop = args.null_prop
    bundle_id = args.bundle_id

    # Extract list of ctimes from bundle database for the given
    # bundle_id - null split combination
    bundle_db = BundleCoordinator.from_dbfile(
        bundle_db, bundle_id=bundle_id, null_prop_val=null_prop
    )
    ctimes = bundle_db.get_ctimes(bundle_id=bundle_id, null_prop_val=null_prop)

    # Extract list of atomic-map metadata (obs_id, wafer, freq_channel)
    # for the observations defined above
    atomic_metadata = []
    db_con = sqlite3.connect(atom_db)
    db_cur = db_con.cursor()
    for ctime in ctimes:
        res = db_cur.execute(
            "SELECT obs_id, wafer FROM atomic WHERE "
            f"freq_channel == '{freq_channel}' AND ctime == '{ctime}'"
        )
        res = res.fetchall()
        for obs_id, wafer in res:
            print("obs_id", obs_id, "wafer", wafer, freq_channel)
            atomic_metadata.append((obs_id, wafer, freq_channel))
    db_con.close()

    # Reading and coadding all atomics with the same sim_id
    for sim_id in range(id_min, id_max+1):
        print(" - sim_id", sim_id)
        coadd_wmap = np.zeros((3, hp.nside2npix(nside)), dtype=np.float32)
        coadd_w = np.zeros((3, hp.nside2npix(nside)), dtype=np.float32)

        # FIXME: Add CAR version

        # HEALPix version
        for id, (obs_id, wafer, freq_channel) in enumerate(atomic_metadata):
            if id % 10 == 0:
                print("    id", id)
            atomic_fname = map_template.format(sim_id=sim_id).replace(
                ".fits",
                f"_obsid{obs_id}_{wafer}_{freq_channel}.fits"
            )
            fname_wmap, fname_w = (
                f"{atomics_dir}/{atomic_fname.replace('.fits', f'_{s}.fits')}"
                for s in ("wmap", "w")
            )
            coadd_wmap += hp.read_map(fname_wmap, field=range(3), nest=True)
            coadd_w += hp.read_map(fname_w, field=range(3), nest=True)
            os.remove(fname_wmap)
            os.remove(fname_w)

        # Setting zero-weight weight pixels to infinity
        coadd_w[coadd_w == 0] = np.inf
        filtered_sim = coadd_wmap / coadd_w

        out_fname = map_template.format(sim_id=sim_id).replace(
            ".fits", f"_bundle{bundle_id}_filtered.fits"
        )
        out_file = f"{out_dir}/{out_fname}"

        # FIXME: CAR version
        # enmap.write_map(out_file, filtered_sim)

        # HEALPix version
        hp.write_map(
            out_file, filtered_sim, dtype=np.float32, overwrite=True,
            nest=True
        )

        for i, f in zip([0, 1, 2], ["I", "Q", "U"]):
            # FIXME: CAR version
            # plot = enplot.plot(
            #    filtered_sim[i],
            #    color="planck",
            #    ticks=10,
            #    range=1.7,
            #    colorbar=True
            # )
            # enplot.write(
            #    f"{plot_dir}/{out_fname.replace('.fits', '')}_{f}",
            #    plot
            # )

            # HEALPix version
            plt.figure()
            hp.mollview(
                filtered_sim[i],
                cmap="RdYlBu_r",
                min=-1.7,
                max=1.7,
                cbar=True,
                nest=True,
                unit=r"$\mu$K"
            )
            plt.savefig(
                f"{plot_dir}/{out_fname.replace('.fits', '')}_{f}.png"
            )
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--atomic-db",
        help="Path to the atomic maps database",
        type=str
    )
    parser.add_argument(
        "--bundle-db",
        help="Path to the bundle database",
        type=str
    )
    parser.add_argument(
        "--map-template",
        help="Template file for the map to filter",
        type=str
    )
    parser.add_argument(
        "--sim-ids",
        help="Comma separated list of simulation ids",
        type=str
    )
    parser.add_argument(
        "--output-directory",
        help="Output directory for the filtered maps",
        type=str
    )
    parser.add_argument(
        "--freq-channel",
        help="Frequency channel to filter",
        type=str
    )
    parser.add_argument(
        "--bundle-id",
        help="Bundle ID to filter",
    )
    parser.add_argument(
        "--null-prop",
        help="Null property to filter",
        default=None
    )
    parser.add_argument(
        "--nside",
        help="Nside parameter for HEALPIX mapmaker",
        type=int, default=512
    )

    args = parser.parse_args()

    main(args)
