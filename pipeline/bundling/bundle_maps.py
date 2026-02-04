import argparse
import bundling_utils as utils
from coadder import Bundler
import os
import numpy as np
from coordinator import BundleCoordinator
import itertools

from procs_pool import get_exec_env

def main(config_file, parallelizor, atomic_list=None, error=True):
    """
    Make bundle db, all bundles, coadd bundles together.

    Parameters
    ----------
    config_file: str
        Path to yaml config
    parallelizor: tuple
        (executor, as_completed_callable, nproc) for MPI/concurrent futures parallelization.
    atomic_list: str
        Filename for atomic list: .npy or .npz of (obs_id, wafer, freq).
    error: bool
        If False errors are caught and printed instead of raised.
    """
    config = utils.Cfg.from_yaml(config_file)

    # Set atomic_list from command line args
    if atomic_list is not None:
        new_map_string = config.map_string_format.replace("{map_type}", atomic_list[:-4] + "_{map_type}")
        config = utils.child_config(config, atomic_list=atomic_list, map_string_format=new_map_string)
        print(f"Set atomic_list to {config.atomic_list}")

    its = [np.atleast_1d(x) for x in [config.freq_channel, config.wafer]]
    patch_list = config.patch_list

    # Main loop over patches
    for patch in patch_list:
        patch_tag = "" if patch is None else patch

        # Make bundle db
        config_db = utils.child_config(config, patch=patch)
        if (not os.path.isfile(config_db.bundle_db_full)) or config_db.overwrite:
            make_bundle_db(config_db)

        if config.only_make_db:
            continue

        # Coadd maps
        else:
            # Loop over wafer/freq combos
            for it in itertools.product(*its):
                freq, wafer = it
                print(patch, freq if freq is not None else "", wafer if wafer is not None else "")
                config_it = utils.child_config(config, patch=patch, freq_channel=freq, wafer=wafer)

                # Science
                if config_it.inter_obs_splits is None and config_it.intra_obs_splits is None:
                    bundle_maps(config_it, config_it.intra_obs_pair, None, parallelizor, error=error)
                # Inter-obs
                if config_it.inter_obs_splits is not None:
                    for inter_obs in np.atleast_1d(config_it.inter_obs_splits):
                        bundle_maps(config_it, config_it.intra_obs_pair, inter_obs, parallelizor, error=error)
                # Intra obs
                if config_it.intra_obs_splits is not None:
                    for intra_obs in config_it.intra_obs_splits:
                        bundle_maps(config_it, intra_obs, None, parallelizor, error=error)

                # Coadd bundles
                coadd_bundles(config_it, wafer, freq, patch_tag, error=error, coadd_fnames=config.save_fnames)

def make_bundle_db(config):
    """
    Make bundle db as determined by config.
    """
    print(f"Writing to {config.bundle_db_full}.")
    bundle_coordinator = BundleCoordinator(
        config.atomic_db, n_bundles=config.n_bundles,
        seed=config.seed, null_props=config.inter_obs_props,
        query_restrict=config.query_restrict_patch,
        atomic_list=config.atomic_list
    )
    bundle_coordinator.save_db(config.bundle_db_full)

def _bundle_maps(config, split_intra_obs=None, split_inter_obs=None, parallelizor=None):
    """
    Make all bundles as determined by config.

    Parameters
    ----------
    config: bundling_utils.Cfg
        config object loaded from yaml
    split_intra_obs: str
        String tag identifying intra obs split to bundle.
    split_inter_obs: str
        String tag identifying inter obs split to bundle.
    parallelizor: tuple
        (executor, as_completed_callable, nproc) for MPI/concurrent futures parallelization.
    """
    out_dir = config.output_dir
    os.makedirs(out_dir, exist_ok=True)

    bundler = Bundler(
        bundle_db=config.bundle_db_full,
        freq_channel=config.freq_channel,
        wafer=config.wafer,
        pix_type=config.pix_type,
        atomic_list=config.atomic_list,
        car_map_template=config.car_map_template,
    )

    bundle_ids = range(config.n_bundles)
    for bundle_id in bundle_ids:
        print(" - bundle_id", bundle_id)

        split_tag = utils.get_split_tag(split_intra_obs, split_inter_obs, config.intra_obs_pair, config.coadd_splits_name)
        wafer_tag = config.wafer if config.wafer is not None else ""
        patch_tag = config.patch if config.patch is not None else ""

        utils.validate_map_string_format(config.map_string_format, wafer_tag, patch_tag)
        out_fname = os.path.join(
            out_dir,
            config.map_string_format.format(split=split_tag,
                                          bundle_id=bundle_id,
                                          wafer=wafer_tag,
                                          patch=patch_tag,
                                          freq_channel=config.freq_channel,
                                          map_type="{}")
        )
        out_fname = out_fname.replace("__", "_")  # Again hacky removal of hopefully accidental double underscores

        if os.path.exists(out_fname.format("map")):
            print(f"Map {out_fname.format('map')} exists: skipping")
            continue

        bundled_map, weights_map, hits_map, fnames = bundler.bundle(
            bundle_id,
            config.map_dir,
            split_label=split_intra_obs,
            null_prop_val=split_inter_obs,
            abscal=config.abscal,
            parallelizor=parallelizor
        )

        fnames = fnames if config.save_fnames else None

        utils.write_maps(out_fname, config.pix_type, bundled_map, weights_map, hits_map, fnames)

        savename_plot = out_fname[:out_fname.find(".fits")] + ".png"
        if config.make_plots:
            utils.plot_map(savename_plot.format("hits"), config.pix_type, hits_map)
            utils.plot_map(savename_plot.format("Q"), config.pix_type, bundled_map[1], unit_fac=1e6, vrange=100)
            utils.plot_map(savename_plot.format("U"), config.pix_type, bundled_map[2], unit_fac=1e6, vrange=100)


def bundle_maps(config, split_intra_obs=None, split_inter_obs=None, parallelizor=None, verbose=True, error=True):
    """See _bundle_maps docstring"""
    split_tag = utils.get_split_tag(split_intra_obs, split_inter_obs, config.intra_obs_pair, config.coadd_splits_name)
    if verbose:
        print(split_tag)
    if error:
        _bundle_maps(config, split_intra_obs, split_inter_obs, parallelizor)
    else:
        try:
            _bundle_maps(config, split_intra_obs, split_inter_obs, parallelizor)
        except ValueError as e:
            print("Error: ", e)

def coadd_bundles(config, wafer, freq, patch_tag, coadd_fnames=False, error=True):
    """Coadd bundles together to get full-split single-bundle or single-split all-bundle coadds."""
    wafer_tag = "" if wafer is None else wafer
    template = os.path.join(config.output_dir, config.map_string_format.format(
        split="{}", bundle_id="{}", wafer=wafer_tag, patch=patch_tag,
        freq_channel=freq, map_type="{}"))
    template = template.replace("__", "_")

    # Make full coadds
    if config.coadd_split_pair is not None:
        print("Making full maps")
        savename = template.format(config.coadd_splits_name, "{}", "{}")
        try:
            utils.make_full(template, config.coadd_split_pair, config.n_bundles, config.pix_type,
                            coadd_hits=True, coadd_fnames=coadd_fnames, savename=savename, return_maps=False)
        except FileNotFoundError as e:
            if error:
                raise e
            else:
                print("Error: ", e)

    if config.coadd_bundles_splitname is not None:
        print("Co-adding bundles")
        for coadd_bundles_splitname in np.atleast_1d(config.coadd_bundles_splitname):
            print(coadd_bundles_splitname)
            temp = template.format(coadd_bundles_splitname, "{}", "{}")
            sum_vals = list(range(config.n_bundles))
            savename = temp.format("!", "{}").replace("_bundle!", "")
            try:
                coadd_map, _, coadd_hits = utils.coadd_bundles(temp, sum_vals, config.pix_type,
                                                               coadd_hits=True, coadd_fnames=coadd_fnames, savename=savename)
                if config.make_plots:
                    savename_plot = savename[:savename.find(".fits")] + ".png"
                    utils.plot_map(savename_plot.format("hits"), config.pix_type, coadd_hits)
                    utils.plot_map(savename_plot.format("mapQ"), config.pix_type, coadd_map[1], unit_fac=1e6, vrange=20)
                    utils.plot_map(savename_plot.format("mapU"), config.pix_type, coadd_map[2], unit_fac=1e6, vrange=20)

            except FileNotFoundError as e:
                if error:
                    raise e
                else:
                    print("Error: ", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make bundled maps")
    parser.add_argument(
        "--config_file", type=str, help="yaml file with configuration."
    )
    parser.add_argument(
        "--atomic_list", type=str, help="atomic list"
    )
    parser.add_argument(
        "--nproc", type=int, default=1, help="Number of parallel processes for concurrent futures."
    )
    parser.add_argument(
        "--error", action="store_true", help="Raise errors instead of catching and printing"
    )

    args = parser.parse_args()
    rank, executor, as_completed_callable = get_exec_env(args.nproc)
    if rank == 0:
        try:
            nproc = executor.num_workers
        except AttributeError:
            nproc = executor._max_workers
        if nproc > 1:
            parallelizor = (executor, as_completed_callable, nproc)
        else:
            parallelizor = None
        main(args.config_file, parallelizor, args.atomic_list, args.error)
