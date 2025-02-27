import numpy as np
import healpy as hp
import argparse
import sqlite3
import os
import sys
import tracemalloc
import matplotlib.pyplot as plt
import sotodlib.preprocess.preprocess_util as pp_util

from pixell import enmap, enplot
from mpi4py import MPI

# TODO: Make it an actual module
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bundling'))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'misc'))
)
from bundling_utils import coadd_maps  # noqa
from coordinator import BundleCoordinator  # noqa
from mpi_utils import distribute_tasks  # noqa


def _get_atomics_maps_list(sim_id, atomic_metadata, freq_channel, atomics_dir,
                           split_label, map_string_format, mfmt=".fits",
                           pix_type="car", remove_atomics=False, logger=None):
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
        logger: sotodlib.preprocess.preprocess_util.logger
            Logger instance to print output.
    Returns:
        wmap_list: list
            List of weighted maps (numpy.ndmap or numpy.ndarray)
        w_list: list
            List of map weights (numpy.ndmap or numpy.ndarray)
    """
    wmap_list, w_list = ([], [])

    for id, (obs_id, wafer) in enumerate(atomic_metadata):
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

    logger.info(f"{id+1} atomics expected, {len(wmap_list)} atomics at disk.")

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


def get_query_atomics(freq_channel, ctimes, split_label="science",
                      query_restrict="median_weight_qu < 2e10"):
    """
    """
    query = f"""
            SELECT obs_id, wafer
            FROM atomic
            WHERE freq_channel == '{freq_channel}'
            AND ctime IN {tuple(ctimes)}
            AND split_label == '{split_label}'
            """
    if split_label != "science":
        # The "valid" argument is not meant to be used with the "science" split
        query += " AND valid == 1"
    elif query_restrict:
        # If we want "science", then we have to use "query_restrict". On the
        # other hand, if we don't use the "science" split, the observations
        # will have been restricted prior to that, so we ignore it there.
        query += f" AND {query_restrict}"
    return query


def main(args):
    """
    """
    # MPI related initialization
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    success = True

    # Initialize the logger
    logger = pp_util.init_logger("benchmark", verbosity=3)

    # Output directories
    out_dir = args.output_directory
    plots_dir = out_dir + "/plots"

    # Path to atomic maps
    atomics_dir = args.atomics_dir

    # Databases
    atom_db = args.atomic_db
    bundle_db = args.bundle_db

    # Sim related arguments
    map_string_format = args.map_string_format
    sim_ids = args.sim_ids
    if "," in sim_ids:
        id_min, id_max = sim_ids.split(",")
        sim_ids = np.arange(int(id_min), int(id_max)+1)
    else:
        sim_ids = np.array([int(sim_ids)])

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

    # Bundle query arguments
    freq_channel = args.freq_channel
    if args.split_labels_inter_obs is None:
        inter_obs_nulls = []
    elif "," in args.split_labels_inter_obs:
        inter_obs_nulls = args.split_labels_inter_obs.split(",")
    else:
        inter_obs_nulls = [args.split_labels_inter_obs]
    bundle_id = args.bundle_id

    # Load all data from bundle db without filtering
    bundle_db = BundleCoordinator.from_dbfile(
        bundle_db, bundle_id=bundle_id
    )

    # Gather all split labels of atomics to be coadded.
    split_labels_coadd, split_labels_nocoadd = ([], [])
    if args.split_labels_intra_obs is not None:
        if "," not in args.split_labels_intra_obs:
            split_labels_nocoadd = [args.split_labels_intra_obs]
        else:
            split_labels_nocoadd = (args.split_labels_intra_obs).split(",")
    if args.split_label_pair_to_coadd is not None:
        if "," not in args.split_label_pair_to_coadd:
            raise ValueError("You must pass a comma-separated string list to "
                             "'--split_label_pair_to_coadd'.")
        else:
            split_labels_coadd = args.split_label_pair_to_coadd.split(",")
    if ((args.split_labels_intra_obs, args.split_label_pair_to_coadd) == (None, None)):  # noqa
        raise ValueError("You must pass at least one of the two: "
                         "'--split_label_pair_to_coadd' "
                         "or '--split_labels_intra_obs'.")

    logger.info(f"Split labels to coadd individually: {split_labels_nocoadd}")  # noqa
    logger.info(f"Split labels to coadd together: {split_labels_coadd}")

    # Extract list of ctimes from bundle database for the given
    # bundle_id - inter obs null label
    ctimes = {
        null_prop_val: bundle_db.get_ctimes(
            bundle_id=bundle_id, null_prop_val=null_prop_val
        ) for null_prop_val in inter_obs_nulls
    }
    # We should add the science split
    ctimes["science"] = bundle_db.get_ctimes(bundle_id=bundle_id)

    for null_prop_val in inter_obs_nulls:
        ctimes[null_prop_val] = [ct for ct in ctimes[null_prop_val] if ct in ctimes["science"]]  # noqa

    # Connect the the atomic map DB
    atomic_metadata = []
    db_con = sqlite3.connect(atom_db)
    db_cur = db_con.cursor()
    query_restrict = args.query_restrict

    # Query all atomics used for science, filtering ctimes
    query = get_query_atomics(
        freq_channel, ctimes["science"], split_label="science",
        query_restrict=query_restrict
    )
    res = db_cur.execute(query)
    res = res.fetchall()
    atomic_metadata = {
        "science": [
            (obs_id, wafer) for obs_id, wafer in res
        ]
    }
    # Query all atomics used for intra-obs splits
    # filtering ctimes and split labels
    for split_label in split_labels_nocoadd:
        query = get_query_atomics(
            freq_channel, ctimes["science"], split_label=split_label,
            query_restrict=query_restrict
        )
        res = db_cur.execute(query)
        res = res.fetchall()
        atomic_metadata[split_label] = [
            (obs_id, wafer) for obs_id, wafer in res
            if (obs_id, wafer) in atomic_metadata["science"]
        ]
    # Query all atomics used for inter-obs splits
    # filtering ctimes w.r.t to the null prop considered
    # for the two intra-obs splits to be coadded
    if len(split_labels_coadd) != 0:
        for null_prop_val in inter_obs_nulls:
            for split_label in split_labels_coadd:
                query = get_query_atomics(
                    freq_channel, ctimes["science"], split_label=split_label
                )
                res = db_cur.execute(query)
                res = res.fetchall()
                atomic_metadata[null_prop_val, split_label] = [
                    (obs_id, wafer) for obs_id, wafer in res
                    if (obs_id, wafer) in atomic_metadata["science"]
                ]
    db_con.close()

    split_labels_all = list(
        set(["science"] + split_labels_nocoadd + inter_obs_nulls)
    )
    mpi_shared_list = [(sim_id, split_label)
                       for sim_id in sim_ids
                       for split_label in split_labels_all]

    task_ids = distribute_tasks(size, rank, len(mpi_shared_list),
                                logger=logger)
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for sim_id, split_label in local_mpi_list:
        map_dir = atomics_dir.format(sim_id=sim_id)
        assert os.path.isdir(map_dir), map_dir

        logger.info(f"Loading atomics for sim {sim_id:04d} and {split_label}")

        w_list, wmap_list = ([], [])

        if split_label == "science":
            tracemalloc.start()
            for sl in split_labels_coadd:
                wmap_l, w_l = _get_atomics_maps_list(
                    sim_id, atomic_metadata[sl],
                    freq_channel, map_dir, sl,
                    map_string_format, mfmt=mfmt, pix_type=pix_type,
                    logger=logger
                )
                wmap_list += wmap_l
                w_list += w_l
            current_gb, peak_gb = [1024**(-3) * c
                                   for c in tracemalloc.get_traced_memory()]
            logger.info("Traced Memory for 'science' (Current, Peak): "
                        f"({current_gb:.2f} GB, {current_gb:.2f} GB")
            tracemalloc.stop()
        elif split_label in inter_obs_nulls:
            for sl in split_labels_coadd:
                wmap_l, w_l = _get_atomics_maps_list(
                    sim_id, atomic_metadata[split_label, sl],
                    freq_channel, map_dir, sl,
                    map_string_format, mfmt=mfmt, pix_type=pix_type,
                    logger=logger
                )
                wmap_list += wmap_l
                w_list += w_l
        else:
            wmap_list, w_list = _get_atomics_maps_list(
                sim_id, atomic_metadata[split_label],
                freq_channel, map_dir, split_label,
                map_string_format, mfmt=mfmt, pix_type=pix_type,
                logger=logger
            )

        logger.info(f"Coadding atomics for sim {sim_id:04d} and {split_label}")
        map_filtered, weights = coadd_maps(
            wmap_list, w_list, pix_type=pix_type,
            car_template_map=car_template_map
        )
        out_fname = map_string_format.format(sim_id=sim_id).replace(
            ".fits",
            f"_bundle{bundle_id}_{freq_channel}_{split_label}_filtered.fits"
        )
        _save_and_plot_map(
            map_filtered, out_fname, out_dir, plots_dir,
            pix_type=pix_type
        )
        _save_and_plot_map(
            weights, out_fname.replace(".fits", "_weights.fits"),
            out_dir, plots_dir, pix_type=pix_type, do_plot=False
        )
    success = all(comm.allgather(success))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--atomic-db",
        help="Path to the atomic maps database.",
        type=str
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
        "--query-restrict",
        help="Query restriction for atomic maps.",
        default=""
    )
    parser.add_argument(
        "--bundle-id",
        type=int,
        help="Bundle ID to filter",
    )
    parser.add_argument(
        "--split_labels_intra_obs",
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
        "--split_labels_inter_obs",
        help="String, or comma-separated string list of inter-obs split "
             "labels, e.g. 'low_pwv', or 'low_pwv,high_sun_distance'. ",
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
