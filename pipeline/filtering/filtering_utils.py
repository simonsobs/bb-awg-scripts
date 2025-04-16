import os
import yaml
import numpy as np
import healpy as hp
from typing import Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

from pixell import enmap, enplot
from sotodlib.coords.demod import make_map
from sotodlib.coords import P


def yaml_loader(config):
    """
    Custom yaml loader to load the configuration file.
    """
    def path_constructor(loader, node):
        return "/".join(loader.construct_sequence(node))
    yaml.SafeLoader.add_constructor("!path", path_constructor)
    with open(config, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def get_atomics_maps_list(sim_id, pure_type, atomic_metadata, freq_channel,
                          atomic_sim_dir, split_label, sim_string_format,
                          mfmt=".fits", pix_type="car", remove_atomics=False,
                          logger=None):
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
        sim_string_format: str
            String format for the filtered atomic maps.
            Must contain {sim_id}, {pure_type}.
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
    if not atomic_metadata:
        print("WARNING: atomic_metadata is empty.")

    for id, (obs_id, wafer) in enumerate(atomic_metadata):
        atomic_fname = sim_string_format.format(sim_id=sim_id,
                                                pure_type=pure_type)
        atomic_fname = atomic_fname.replace(
            mfmt,
            f"_{obs_id}_{wafer}_{freq_channel}_{split_label}{mfmt}"
        )
        fname_wmap, fname_w = (
            f"{atomic_sim_dir}/{atomic_fname.replace(mfmt, f'_{s}{mfmt}')}"
            for s in ("wmap", "weights")
        )

        # Observations can vanish if the FP thinning and the detector cuts
        # conspire such that no detectors are left. Since this is a rare
        # case, it is acceptable to just ignore those when coadding.
        # print("expected file:", fname_wmap)
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


def save_and_plot_map(map, out_fname, out_dir, plot_dir, pix_type="car",
                      overwrite=False, do_plot=True):
    """
    Saves and optionally plots TQU map.
    """
    if os.path.isfile(f"{out_dir}/{out_fname}"):
        print(" FILE EXISTS: {out_dir}/{out_fname}")
        return
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
                f"{plot_dir}/{out_fname.replace('.fits', '')}_{f}", plot
            )

        elif pix_type == "hp":
            plt.figure()
            hp.mollview(
                map[i], cmap="RdYlBu_r", min=-1.7, max=1.7,
                cbar=True, nest=True, unit=r"$\mu$K"
            )
            plt.savefig(
                f"{plot_dir}/{out_fname.replace('.fits', '')}_{f}.png"
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


def get_fullsky_geometry(res_arcmin=5., variant="fejer1"):
    """
    Generates a fullsky CAR template at resolution res-arcmin.
    """
    res = res_arcmin * np.pi/180/60
    return enmap.fullsky_geometry(res=res, proj='car', variant=variant)


def make_map_wrapper(obs, split_labels, pix_type="hp", shape=None, wcs=None,
                     nside=None, site=None, logger=None):
    """
    """
    obs.wrap("weather", np.full(1, "toco"))
    obs.wrap("site", np.full(1, site))
    if pix_type == "car":
        nside = None
        assert wcs is not None
    elif pix_type == "hp":
        wcs = None
        assert nside is not None

    inv_var = 1 / obs.preprocess.noiseQ_nofit.white_noise ** 2

    wmap_dict = {}
    weights_dict = {}

    for split_label in split_labels:
        cuts = obs.flags.glitch_flags + ~obs.preprocess.split_flags.cuts[split_label]  # noqa

        Proj = P.for_tod(obs, wcs_kernel=wcs, comps='TQU', cuts=cuts,
                         hwp=True, interpol=None)
        result = make_map(obs, P=Proj, det_weights=2 * inv_var,
                          det_weights_demod=inv_var)
        wmap_dict[split_label] = result['weighted_map']
        weights_dict[split_label] = result['weight']
        # transform (3, 3, N, n) array to (3, N, n) keeping only diagonals
        # in the first two dimensions
        weights_dict[split_label] = np.moveaxis(
            weights_dict[split_label].diagonal(), -1, 0
        )

    return wmap_dict, weights_dict


@dataclass
class Cfg:
    """
    Class to configure filtering

    Args
    --------
    bundle_db: str
        Path to bundling database
    atomic_db: str
        Path to atomic map database
    preprocess_config_init: str
        Path to preprocessing init yaml file
    preprocess_config_proc: str
        Path to preprocessing proc yaml file
    query_restrict: str
        SQL query to restrict obs from the atomic database
    pix_type: str
        'hp' or 'car'
    sim_dir: str
        Path to directory containing unfiltered input sims
    atomic_sim_dir: str
        Path to directory containing filtered atomic sims
    output_dir: str
        Path to output directory
    sim_ids: list
        Simulation seeds to be filtered, passed as integers
    bundle_id: int
        Bundle ID to be filtered
    sim_string_format: str
        String formatting for unfiltered input sims
        must contain {sim_id} and {pure_type}.
    freq_channel: str
        Frequency channel, e.g. 'f090'.
    intra_obs_splits: list
        List of split labels for intra-obs splits, e.g. 'scan_left'.
    intra_obs_pair: list
        Pair of intra-obs labels that will be added to make full obs
        for inter-obs splits
    inter_obs_splits:
        List of inter-obs split names for which to create bundles
    car_map_template: str
        Path to CAR map or geometry to be used as template
    nside: int
        HEALPix NSIDE parameter
    fp_thin: int
        Focal plane thinning factor applied to the sim filtering
    nbatch_atomics: int
        Number of batches to divide the bundle into, based on random timestamp
        splits
    """
    pwg_scripts_dir: str
    bb_awg_scripts_dir: str
    bundle_db: str
    atomic_db: str
    preprocess_config_init: str
    preprocess_config_proc: str
    sim_dir: str
    atomic_sim_dir: str
    output_dir: str
    sim_ids: list
    sim_string_format: str
    freq_channel: str
    intra_obs_splits: list
    query_restrict: Optional[str] = ""
    pix_type: Optional[str] = "car"
    bundle_id: Optional[int] = 0
    intra_obs_pair: Optional[list] = None
    inter_obs_splits: Optional[list] = None
    car_map_template: Optional[str] = None
    nside: Optional[int] = None
    fp_thin: Optional[int] = 8
    nbatch_atomics: Optional[int] = None

    def __post_init__(self):
        # Add extra defaults for private args not expected in config file
        pass

    @classmethod
    def from_yaml(cls, path) -> "Cfg":
        d = yaml_loader(path)
        return cls(**d)
