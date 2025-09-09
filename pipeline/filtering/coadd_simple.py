import numpy as np
import argparse
import sqlite3
import os
import sys

import sotodlib.preprocess.preprocess_util as pp_util

# TODO: Make it an actual module
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bundling'))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'misc'))
)
import mpi_utils as mpi # noqa
import bundling_utils as bu  # noqa
import filtering_utils as fu  # noqa
import coordinator as coord  # noqa


def main(args):
    if args.pix_type not in ["hp", "car"]:
        raise ValueError(
            "Unknown pixel type, must be 'car' or 'hp'."
        )
    
    # Initialize the logger
    logger = pp_util.init_logger("summary", verbosity=1)

    for required_tag in ["{sim_id"]:
        if required_tag not in args.sim_string_format:
            logger.warning("sim_string_format does not have "
                           f"required placeholder {required_tag}")

    # MPI related initialization
    rank, size, comm = mpi.init(True, logger=logger)

    # Output directories
    out_dir = args.output_dir
    coadded_dir = out_dir + "/coadded_sims"
    plot_dir = out_dir + "/plots"
    os.makedirs(coadded_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Path to atomic maps
    atomic_sim_dir = args.atomic_sim_dir

    # Databases
    atom_db = args.atomic_db

    # Sim related arguments
    sim_string_format = args.sim_string_format
    sim_ids = args.sim_ids
    pure_types = args.pure_types

    if isinstance(sim_ids, str):
        if "," in sim_ids:
            id_min, id_max = sim_ids.split(",")
            sim_ids = np.arange(int(id_min), int(id_max)+1)
        else:
            sim_ids = np.array([int(sim_ids)])

    # Pixelization arguments
    pix_type = args.pix_type
    if pix_type == "hp":
        mfmt = ".fits"  # TODO: test fits.gz for HEALPix
        car_map_template = None
    elif pix_type == "car":
        mfmt = ".fits"
        car_map_template = args.car_map_template

    # Connect the the atomic map DB
    db_con = sqlite3.connect(atom_db)
    db_cur = db_con.cursor()
    query_restrict = args.query_restrict

    # Query all atomics used for null splits in intra_obs_pair
    intra_obs_pair = args.intra_obs_pair
    #intra_obs_pair = ["science"]
    freq_channel = args.freq_channel  # we might wanna allow for general input

    atomic_metadata = {}
    for split_label in intra_obs_pair:
        query = f"""
                SELECT obs_id, wafer
                FROM atomic
                WHERE freq_channel == '{freq_channel}'
                AND (split_label == '{split_label}')
                AND {query_restrict}
                """
        res = db_cur.execute(query)
        res = res.fetchall()
        atomic_metadata[split_label] = [(obs_id, wafer)
                                        for obs_id, wafer in res]
    #atomic_metadata = [(obs_id, wafer) for obs_id, wafer in res]
    db_con.close()

    mpi_shared_list = [(sim_id, pure_type)
                       for sim_id in sim_ids
                       for pure_type in [f"pure{i}" for i in pure_types]]

    # Every rank must have the same shared list
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list),
                                    logger=logger)
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for sim_id, pure_type in local_mpi_list:
        logger.info(sim_id, pure_type)
        map_dir = atomic_sim_dir.format(sim_id=sim_id)
        assert os.path.isdir(map_dir), map_dir
        out_fname = sim_string_format.format(sim_id=sim_id,
                                             pure_type=pure_type)
        out_fname = out_fname.replace(
            ".fits",
            f"_{freq_channel}_science_filtered.fits"
        )
        if os.path.isfile(f"{coadded_dir}/{out_fname}"):
            logger.warning(f"Sim {sim_id}, {pure_type}: File exists, ignoring.")
            return

        wmap_list, w_list = ([], [])
        for split_label in intra_obs_pair:
            logger.info(split_label, "len(atomic_metadata):",
                        len(atomic_metadata[split_label]))
            wmap_l, w_l = fu.get_atomics_maps_list(
                sim_id, pure_type, atomic_metadata[split_label],
                freq_channel, map_dir, split_label,
                sim_string_format, mfmt=mfmt, pix_type=pix_type,
                remove_atomics=True,  # CHANGED: Remove all atomics
                logger=logger
            )
            wmap_list += wmap_l
            w_list += w_l
        map_filtered, weights = bu.coadd_maps(
            wmap_list, w_list, pix_type=pix_type,
            car_template_map=car_map_template
        )
        logger.info("Plot dir:", plot_dir)
        fu.save_and_plot_map(
            map_filtered, out_fname, coadded_dir, plot_dir,
            pix_type=pix_type, logger=logger
        )
        fu.save_and_plot_map(
            weights, out_fname.replace(".fits", "_weights.fits"),
            coadded_dir, plot_dir, pix_type=pix_type, do_plot=False,
            logger=logger
        )
    comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, help="yaml file with configuration."
    )

    args = parser.parse_args()
    config = fu.Cfg.from_yaml(args.config_file)
    main(config)
