import numpy as np
import healpy as hp
import argparse
import sqlite3
import os
import sys
import matplotlib.pyplot as plt

from pixell import enmap, enplot

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bundling'))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'misc'))
)
from bundling_utils import coadd_maps  # noqa
from coordinator import BundleCoordinator  # noqa


def _get_atomics_maps_list(sim_id, atomic_metadata, atomics_dir, split_label,
                           map_string_format, mfmt=".fits", pix_type="car",
                           remove_atomics=False):
    """
    Returns a list of filtered atomic maps that correpsond to a given
    simulation ID, given a list of atomic metadata.

    Parameters:
        sim_id: int
            Simulation ID.
        atomic_metadata: list
            List of tuples if strings (obs_id, wafer, freq_channel).
        split_label: str
            Map string label corresponding to the split, e.g. 'det_left'
        map_string_format: str
            String format for the filtered atomic maps; must contain {sim_id}.
        mfmt: str
            Atomic file name ending.
        pix_type: str
            Pixelization type; either 'car' or 'hp'.
        remove_atomics: bool
            Whether to remove atomic map files after loading into list.
    Returns:
        wmap_list: list
            List of weighted maps (numpy.ndmap or numpy.ndarray)
        w_list: list
            List of map weights (numpy.ndmap or numpy.ndarray)
    """
    wmap_list = []
    w_list = []

    for id, (obs_id, wafer, freq_channel) in enumerate(atomic_metadata):
        if id % 10 == 0:
            print("    id", id)
        atomic_fname = map_string_format.format(sim_id=sim_id).replace(
            mfmt,
            f"_{obs_id}_{wafer}_{freq_channel}_{split_label}{mfmt}"
        )
        fname_wmap, fname_w = (
            f"{atomics_dir}/{atomic_fname.replace(mfmt, f'_{s}{mfmt}')}"
            for s in ("wmap", "weights")
        )

        # Observations can vanish if the FP thinning and the detector cuts
        # conspire such that no detectors are left. Since this is a rare
        # case, it is acceptable to just ignore those when coadding.
        if not (os.path.isfile(fname_wmap) and os.path.isfile(fname_w)):
            print("WARNING: "
                  f"{fname_wmap}"
                  "is missing. SKIPPING in coadder.")
            continue

        if pix_type == "car":
            wmap = enmap.read_map(fname_wmap)
            w = enmap.read_map(fname_w)
        elif pix_type == "hp":
            wmap = hp.read_map(fname_wmap, field=range(3), nest=True)
            w = hp.read_map(fname_w, field=range(3), nest=True)

        wmap_list.append(wmap)
        w_list.append(w)

        if remove_atomics:
            os.remove(fname_wmap)
            os.remove(fname_w)

    return wmap_list, w_list


def _save_and_plot_map(map, out_fname, out_dir, plots_dir, pix_type="car",
                       do_plot=True):
    """
    Saves and optionally plots TQU map.
    """
    if pix_type == "car":
        enmap.write_map(f"{out_dir}/{out_fname}", map)

    elif pix_type == "hp":
        hp.write_map(
            f"{out_dir}/{out_fname}", map, dtype=np.float64, overwrite=True,
            nest=True
        )
    if not do_plot:
        return

    for i, f in zip([0, 1, 2], ["I", "Q", "U"]):
        if pix_type == "car":
            if isinstance(map, tuple):
                map = map[0]  # For enmap.ndmaps
            plot = enplot.plot(
                map[i], color="planck", ticks=10, range=1.7, colorbar=True
            )
            enplot.write(
                f"{plots_dir}/{out_fname.replace('.fits', '')}_{f}", plot
            )

        elif pix_type == "hp":
            plt.figure()
            hp.mollview(
                map[i], cmap="RdYlBu_r", min=-1.7, max=1.7,
                cbar=True, nest=True, unit=r"$\mu$K"
            )
            plt.savefig(
                f"{plots_dir}/{out_fname.replace('.fits', '')}_{f}.png"
            )
            plt.close()


def main(args):
    """
    """
    # ArgumentParser
    out_dir = args.output_directory
    plots_dir = f"{out_dir}/plots"
    if not os.path.isdir(f"{out_dir}/plots"):
        raise ValueError(f"Directory does not exist: {out_dir}/plots")
    atomics_dir = args.atomics_dir

    # Databases
    atom_db = args.atomic_db
    bundle_db = args.bundle_db

    # Sim related arguments
    map_string_format = args.map_string_format
    sim_ids = args.sim_ids

    # Pixelization arguments
    pix_type = args.pix_type
    if pix_type == "hp":
        # nside = args.nside
        # ordering_hp = "RING"
        mfmt = ".fits"  # TODO: fits.gz for HEALPix
        car_template_map = None
    elif pix_type == "car":
        # nside = None
        # ordering_hp = None
        mfmt = ".fits"
        car_template_map = args.car_template_map

    if "," in sim_ids:
        id_min, id_max = sim_ids.split(",")
    else:
        id_min = sim_ids
        id_max = id_min
    id_min, id_max = (int(id_min), int(id_max))

    # Bundle query arguments
    freq_channel = args.freq_channel
    null_prop_val_inter_obs = args.null_prop_val_inter_obs
    bundle_id = args.bundle_id

    bundle_db = BundleCoordinator.from_dbfile(
        bundle_db, bundle_id=bundle_id, null_prop_val=null_prop_val_inter_obs
    )

    # Gather all split labels of atomics to be coadded.
    split_labels_coadd, split_labels_nocoadd = ([], [])
    if args.split_label_intra_obs is not None:
        if "," not in args.split_label_intra_obs:
            split_labels_nocoadd = [args.split_label_intra_obs]
        else:
            split_labels_nocoadd = (args.split_label_intra_obs).split(",")
    if args.split_label_pair_to_coadd is not None:
        if "," not in args.split_label_pair_to_coadd:
            raise ValueError("You must pass a comma-separated string list to "
                             "'--split_label_pair_to_coadd'.")
        else:
            split_labels_coadd = args.split_label_pair_to_coadd.split(",")
    if ((args.split_label_intra_obs, args.split_label_pair_to_coadd) == (None, None)):  # noqa
        raise ValueError("You must pass at least one of the two: "
                         "'--split_label_pair_to_coadd' "
                         "or '--split_label_intra_obs'.")

    print(" - split labels to individually coadd:", split_labels_nocoadd)
    print(" - split labels to coadd together: ", split_labels_coadd)

    # Extract list of ctimes from bundle database for the given
    # bundle_id - null split combination
    ctimes_dict = {
        label: bundle_db.get_ctimes(
            bundle_id=bundle_id, split_label=label,
            null_prop_val=null_prop_val_inter_obs
        ) for label in list(set(split_labels_coadd + split_labels_nocoadd))
    }

    # Read restrictive list of atomic-map metadata
    # (obs_id, wafer, freq_channel) from file, and intersect it
    # with the metadata in the bundling database.
    atomic_restrict = []
    if args.atomic_list is not None:
        atomic_restrict = list(
            map(tuple, np.load(args.atomic_list)["atomic_list"])
        )

    # Extract list of atomic-map metadata (obs_id, wafer, freq_channel)
    # for the observations defined above
    atomic_metadata_dict = {
        label: []
        for label in list(set(split_labels_coadd + split_labels_nocoadd))
    }

    db_con = sqlite3.connect(atom_db)
    db_cur = db_con.cursor()

    for split_label in ctimes_dict:
        atomic_metadata = atomic_metadata_dict[split_label]
        for ctime in ctimes_dict[split_label]:
            res = db_cur.execute(
                "SELECT obs_id, wafer FROM atomic WHERE "
                f"freq_channel == '{freq_channel}' AND ctime == '{ctime}' "
                f" AND split_label == '{split_label}'"
            )
            res = res.fetchall()

            for obs_id, wafer in res:
                atom_id = (obs_id, wafer, freq_channel)

                do_include = (args.atomic_list is None
                              or (atom_id in atomic_restrict))
                if do_include and not (atom_id in atomic_metadata):
                    atomic_metadata.append((obs_id, wafer, freq_channel))
    db_con.close()

    # OLD VERSION
    # wmap_list = []
    # w_list = []
    # # Reading and coadding all atomics with the same sim_id
    # for sim_id in range(id_min, id_max+1):
    #     print(" - sim_id", sim_id)
    #     coadd_wmap = np.zeros((3, hp.nside2npix(nside)), dtype=np.float32)
    #     coadd_w = np.zeros((3, hp.nside2npix(nside)), dtype=np.float32)

    #     # FIXME: Add CAR version

    #     # HEALPix version
    #     for id, (obs_id, wafer, freq_channel) in enumerate(atomic_metadata):
    #         if id % 10 == 0:
    #             print("    id", id)
    #         atomic_fname = map_string_format.format(sim_id=sim_id).replace(
    #             ".fits",
    #             f"_obsid{obs_id}_{wafer}_{freq_channel}.fits"
    #         )
    #         fname_wmap, fname_w = (
    #             f"{atomics_dir}/{atomic_fname.replace('.fits', f'_{s}.fits')}"  # noqa
    #             for s in ("wmap", "w")
    #         )
    #         coadd_wmap += hp.read_map(fname_wmap, field=range(3), nest=True)
    #         coadd_w += hp.read_map(fname_w, field=range(3), nest=True)
    #         os.remove(fname_wmap)
    #         os.remove(fname_w)

    #     # Setting zero-weight weight pixels to infinity
    #     coadd_w[coadd_w == 0] = np.inf
    #     filtered_sim = coadd_wmap / coadd_w

    print("## Reading atomics ##")
    for sim_id in range(id_min, id_max+1):
        print(" - sim_id", sim_id)

        # Coadding atomics for individual splits
        for split_label in split_labels_nocoadd:
            wmap_list, w_list = _get_atomics_maps_list(
                sim_id, atomic_metadata, atomics_dir, split_label,
                map_string_format, mfmt=mfmt, pix_type=pix_type
            )
            print(f"## Coadding atomics ({split_label}) ##")
            filtered_sim = coadd_maps(
                wmap_list, w_list, pix_type=pix_type,
                car_template_map=car_template_map
            )
            del wmap_list, w_list
            out_fname = map_string_format.format(sim_id=sim_id).replace(
                ".fits", f"_bundle{bundle_id}_{split_label}_filtered.fits"
            )
            _save_and_plot_map(
                filtered_sim, out_fname, out_dir, plots_dir, pix_type=pix_type
            )

        # Coadding atomics to get science split
        wmap_list, w_list = ([], [])
        for split_label in split_labels_coadd:
            wmap_l, w_l = _get_atomics_maps_list(
                sim_id, atomic_metadata, atomics_dir, split_label,
                map_string_format, mfmt=mfmt, pix_type=pix_type
            )
            wmap_list += wmap_l
            w_list += w_l

        print("## Coadding atomics (science) ##")
        filtered_sim = coadd_maps(
            wmap_list, w_list, pix_type=pix_type,
            car_template_map=car_template_map
        )
        del wmap_list, w_list
        out_fname = map_string_format.format(sim_id=sim_id).replace(
            ".fits", f"_bundle{bundle_id}_science_filtered.fits"
        )
        _save_and_plot_map(
            filtered_sim, out_fname, out_dir, plots_dir, pix_type=pix_type
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--atomic-db",
        help="Path to the atomic maps database.",
        type=str
    )
    parser.add_argument(
        "--atomic_list",
        help="Npz file with list of atomic maps to restrict the atomic_db to.",
        default=None
    )
    parser.add_argument(
        "--bundle-db",
        help="Path to the bundling database.",
        type=str
    )
    parser.add_argument(
        "--map_string_format",
        help="String formatting; must contain {sim_id}."
    )
    parser.add_argument(
        "--sim-ids",
        help="String of format 'sim_id_min,sim_id_max', or only 'sim_id'."
    )
    parser.add_argument(
        "--output-directory",
        help="Output directory for the filtered maps."
    )
    parser.add_argument(
        "--atomics_dir",
        help="Directory to read atomic maps from."
    )
    parser.add_argument(
        "--freq-channel",
        help="Frequency channel to filter",
        type=str
    )
    parser.add_argument(
        "--bundle-id",
        type=int,
        help="Bundle ID to filter",
    )
    parser.add_argument(
        "--null_prop_val_inter_obs",
        help="Null property value for inter-obs splits, e.g. 'pwv_low'.",
        default=None
    )
    parser.add_argument(
        "--split_label_intra_obs",
        help="String, or comma-separated string list of intra-obs split "
             "labels, e.g. 'scan_left', or 'scan_left,det_in,det_left'. "
             "For each label passed, the corresponding atomic maps are "
             "coadded into a split-wise bundle.",
        default=None
    )
    parser.add_argument(
        "--split_label_pair_to_coadd",
        help="Pair of split labels, comma-separated, to be coadded into a "
             "science bundle, e.g. 'scan_left,scan_right. The resulting "
             "bundle map carries the new label 'science'.",
        default=None
    )
    parser.add_argument(
        "--pix_type",
        help="Pixelization type for maps: 'hp' or 'car'.",
        default="hp"
    )
    parser.add_argument(
        "--nside",
        help="Nside parameter for HEALPIX mapmaker.",
        type=int,
        default=512
    )
    parser.add_argument(
        "--car_template_map",
        default=None,
        help="CAR map used to get the geometry for the coadded CAR maps."
    )

    args = parser.parse_args()

    main(args)
