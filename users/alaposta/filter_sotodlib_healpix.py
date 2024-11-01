import sotodlib.mapmaking.demod_mapmaker as dmm                                                                                                                                      
from sotodlib.mapmaking.noise_model import NmatUnit
from sotodlib.site_pipeline import preprocess_tod
from pixell import enmap, utils, enplot
from sotodlib.core import Context
from sotodlib.coords import demod
import matplotlib.pyplot as plt
from mpi4py import MPI
import healpy as hp
import numpy as np
import argparse
import logging
import yaml
import os
import re
import so3g


def erik_make_map(obs, shape=None, wcs=None, nside=None, site=None):
    """
    """
    obs.wrap("weather", np.full(1, "toco"))                                                                                                                                                                    
    obs.wrap("site",    np.full(1, site))
    obs.flags.wrap('glitch_flags', obs.preprocess.turnaround_flags.turnarounds 
                       + obs.preprocess.jumps_2pi.jump_flag + obs.preprocess.glitches.glitch_flags, )
    mapmaker = dmm.setup_demod_map(NmatUnit(), shape=shape, wcs=wcs, nside=nside)
    mapmaker.add_obs('signal', obs)
    wmap = mapmaker.signals[0].rhs[0]
    weights = np.diagonal(mapmaker.signals[0].div[0], axis1=0, axis2=1)
    weights = np.moveaxis(weights, -1, 0)

    return wmap, weights


def get_logger(fmt=None, datefmt=None, debug=False, **kwargs):
    """Return logger from logging module
    code from pspipe

    Parameters
    ----------

    fmt: string
      the format string that preceeds any logging message
    datefmt: string
      the date format string
    debug: bool
      debug flag
    """
    fmt = fmt or "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s" # noqa
    datefmt = datefmt or "%d-%b-%y %H:%M:%S"
    logging.basicConfig(
        format=fmt,
        datefmt=datefmt,
        level=logging.DEBUG if debug else logging.INFO,
        force=True
    )
    return logging.getLogger(kwargs.get("name"))


def main(args):
    """
    """

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # ArgumentParser
    out_dir = args.output_directory
    plot_dir = f"{out_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    atomic_maps_list = args.atomic_maps_list
    preprocess_config = args.preprocess_config
    map_dir = args.map_dir
    map_template = args.map_template
    sim_ids = args.sim_ids

    os.makedirs(out_dir, exist_ok=True)
    with open(atomic_maps_list, "r") as f:
        atomic_maps_list = f.read().splitlines()
    atomic_maps_list = atomic_maps_list

    config = yaml.safe_load(open(preprocess_config, "r"))
    context = config["context_file"]
    ctx = Context(context)

    atomic_metadata = []
    for atom in atomic_maps_list:
        map_id = re.findall(r"[0-9]{10}", atom)[0]
        wafer = re.findall(r"ws[0-9]{1}", atom)[0]
        freq = re.findall(r"f[0-9]{3}", atom)[0]

        query = ctx.obsdb.query(f"cast(timestamp as int) == '{map_id}'")
        obs_id = query[0]["obs_id"]
        atomic_metadata.append((obs_id, wafer, freq))

    metas = {}
    for obs_id, wafer, freq in atomic_metadata:
        dets = {"wafer_slot": wafer, "wafer.bandpass": freq}
        meta = ctx.get_meta(obs_id, dets=dets)
        # Missing pointing not cut in preprocessing
        meta.restrict(
            "dets",
            meta.dets.vals[~np.isnan(meta.focal_plane.gamma)]
        )
        metas[obs_id, wafer, freq] = meta

    id_min, id_max = sim_ids.split(",")
    id_min = int(id_min)
    id_max = int(id_max)

    ids = np.arange(id_min, id_max+1)
    mpi_shared_list = [
        (i, j) for i in ids for j in atomic_metadata
    ]

    if size > len(mpi_shared_list):
        local_start = rank
        local_stop = rank + 1

    else:
        local_start = rank * (len(mpi_shared_list) // size)
        local_stop = local_start + (len(mpi_shared_list) // size)

    log = get_logger()

    local_mpi_list = mpi_shared_list[local_start:local_stop]

    if rank >= len(mpi_shared_list):
        local_mpi_list = []

    leftover = mpi_shared_list[-(len(mpi_shared_list) % size):]
    if rank < len(leftover):
        local_mpi_list.append(leftover[rank])

    log.info(f"Rank {rank} has {len(local_mpi_list)} {[l[0] for l in local_mpi_list]}") # noqa
    log.info(f"Total mpi list is {len(mpi_shared_list)}")
    log.info(f"Total mpi list is {[l[0] for l in mpi_shared_list]}")
    local_wmaps = []
    local_weights = []
    local_labels = []

    for sim_id, (obs_id, wafer, freq) in local_mpi_list:

        map_fname = map_template.format(sim_id=sim_id)
        map_file = f"{map_dir}/{map_fname}"

        # CAR version
        # sim = enmap.read_map(map_file)
        # HEALPix version
        sim = hp.read_map(map_file, field=[0, 1, 2])

        log.info(f"***** Doing {obs_id} {wafer} {freq} and SIMULATION {sim_id} *****") # noqa
        #try:
        aman = preprocess_tod.load_preprocess_tod_sim(
                            obs_id,
                            sim_map=sim,
                            configs=config,
                            meta=metas[obs_id, wafer, freq],
                            modulated=True,
                            site="so_sat1",  # new field required from new from map function
                            ordering="RING"  # new field required for healpix
                        )
        log.info(f"Loaded {obs_id}, {wafer}, {freq}")

        #except: # noqa
        #    log.info(f"Failed to load {obs_id}, {wafer}, {freq}")
        #    continue

        if aman.dets.count <= 1:
            continue
        
        # CAR version
        # filtered_sim = demod.make_map(
        #     aman,
        #     res=10*utils.arcmin,
        #     wcs_kernel=sim.wcs,
        # )
        # wmap, w = filtered_sim["weighted_map"], filtered_sim["weight"]
        # w = np.moveaxis(w.diagonal(), -1, 0)
        # HEALPix version
        wmap, w = erik_make_map(
            aman,
            nside=256,
            site="so_sat1"
        )

        local_wmaps.append(wmap)
        local_weights.append(w)
        local_labels.append(sim_id)

    if rank != 0:
        log.info(f"Rank {rank} sending labels {local_labels}")
        comm.send(local_wmaps, dest=0, tag=0)
        comm.send(local_weights, dest=0, tag=1)
        comm.send(local_labels, dest=0, tag=2)
    else:
        out_labs = []
        out_wmaps = []
        out_weights = []
        for src_rank in range(0, size):
            if src_rank == 0:
                out_labs += local_labels
                out_wmaps += local_wmaps
                out_weights += local_weights
            else:
                wmaps = comm.recv(source=src_rank, tag=0)
                weights = comm.recv(source=src_rank, tag=1)
                labels = comm.recv(source=src_rank, tag=2)
                log.info(f"Rank {src_rank} sent labels {labels}")

                out_labs += labels
                out_wmaps += wmaps
                out_weights += weights

        log.info(out_labs)
        for i in range(id_min, id_max+1):
            log.info(f"Count for sim_id={i} is {out_labs.count(i)}")

        templates = {
            sim_id: [sim.copy() * 0., sim.copy() * 0.]
            for sim_id in range(id_min, id_max+1)
        }

        for index, sim_id in enumerate(out_labs):
            wmap = out_wmaps[index]
            w = out_weights[index]
            # CAR version
            # templates[sim_id][0] = enmap.insert(
            #    templates[sim_id][0],
            #    wmap,
            #    op=np.ndarray.__iadd__
            # )
            # templates[sim_id][1] = enmap.insert(
            #    templates[sim_id][1],
            #    w,
            #    op=np.ndarray.__iadd__
            # )
            # HEALPix version
            templates[sim_id][0] += wmap
            templates[sim_id][1] += w

        for sim_id in range(id_min, id_max + 1):
            templates[sim_id][1][templates[sim_id][1] == 0] = np.inf
            filtered_sim = templates[sim_id][0] / templates[sim_id][1]

            out_fname = map_template.format(sim_id=sim_id).replace(
                ".fits",
                "_filtered.fits"
            )
            out_file = f"{out_dir}/{out_fname}"

            # CAR version
            # enmap.write_map(out_file, filtered_sim)
            # HEALPix version
            hp.write_map(out_file, filtered_sim, overwrite=True, nest=True)

            for i, f in zip([0, 1, 2], ["I", "Q", "U"]):
                # CAR version
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
                    nest=True
                )
                plt.savefig(
                    f"{plot_dir}/{out_fname.replace('.fits', '')}_{f}.png"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--atomic-maps-list",
                        help="List of obs ids included")
    parser.add_argument("--preprocess-config",
                        help="Preprocess config file")
    parser.add_argument("--map-dir",
                        help="Map to filter")
    parser.add_argument("--map-template",
                        help="Map template")
    parser.add_argument("--sim-ids",
                        help="Sim id")
    parser.add_argument("--output-directory",
                        help="Name of the filtered map directory")

    args = parser.parse_args()

    main(args)
