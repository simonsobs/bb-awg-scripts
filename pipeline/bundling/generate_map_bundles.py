import os
import numpy as np
import argparse
from pixell import enmap
import healpy as hp
import random
import coordinator
import coadder
import sys

sys.path.insert(1, '../../pipeline/misc')
import mpi_utils as mpi  # noqa


def _coadd_maps_car(map_list, sign_list=None, res=10, dec_cut=(-75, 25),
                    return_map_weights=True):
    """
    Taken from soopercool/soopercool/map_utils.py
    Reads and coadds CAR maps on a rectangular base map template and outputs
    coadded TQU signal map, TQU weights, and hits map.

    Parameters
    ----------
        map_list: list
            List of strings with names of weighted maps ending with "wmap.fits"
        sign_list: list
            List of numbers corresponding to the sign to multiply the
            respective map before coading. If None, assume list of ones.
        res: int or float
            Angular resolution of CAR map in arcmin. Default: 10
        dec_cut: tuple of float or int
            Declination cuts indegrees defining the CAR geometry.
            Default: (-75, 25)
        return_atomic_weights: bool
            Whether to return the pixel-averaged polarization map weight of
            of each atomic.

    Returns
    -------
        signal: array-like
            Coadded signal map, shape (3, npix_ra, npix_dec).
        weights: array-like
            Coadded weights map, shape (3, npix_ra, npix_dec).
        hits: array-like
            Coadded hits map, shape (npix_ra, npix_dec).
        atomic_weights: list
            List of polarization map weights per atomic map.
    """
    template_geom = enmap.band_geometry(
        (np.deg2rad(dec_cut[0]), np.deg2rad(dec_cut[1])),
        res=np.deg2rad(res/60)
    )
    wmap = enmap.zeros((3, *template_geom[0]), template_geom[1])
    weights = enmap.zeros((3, *template_geom[0]), template_geom[1])
    hits = enmap.zeros(*template_geom)
    atomic_map_weights = []

    if sign_list is None:
        sign_list = [1.]*len(map_list)
    else:
        assert len(sign_list) == len(map_list)

    # Read and coadd weighted maps, weights and hits.
    for f, s in zip(map_list, sign_list):
        m = float(s) * enmap.read_map(f)
        wmap = enmap.insert(wmap, m, op=np.ndarray.__iadd__)
        w = enmap.read_map(f.replace("wmap", "weights"))
        w = np.moveaxis(w.diagonal(), -1, 0)
        weights = enmap.insert(weights, w, op=np.ndarray.__iadd__)
        h = enmap.read_map(f.replace("wmap", "hits"))
        if len(h.shape) == 3:
            h = h[0]
        hits = enmap.insert(hits, h, op=np.ndarray.__iadd__)
        atomic_map_weights.append(0.5 * np.sum(w[1:, :, :], axis=(0, 1, 2)))

    # Cut pixels with nonpositive weights
    weights[weights <= 0] = np.inf
    signal = wmap / weights

    if return_map_weights:
        return signal, weights, hits, np.array(atomic_map_weights)
    return signal, weights, hits


def _coadd_maps_hp(map_list, sign_list=None, res=None, dec_cut=None,
                   return_map_weights=True):
    """
    """
    npix = len(hp.read_map(map_list[0]))
    wmap = np.zeros((3, npix))
    weights = np.zeros((3, npix))
    hits = np.zeros(npix)
    atomic_map_weights = []

    if sign_list is None:
        sign_list = [1.]*len(map_list)
    else:
        assert len(sign_list) == len(map_list)

    for f, s in zip(map_list, sign_list):
        m = float(s) * hp.read_map(f, field=range(3))
        w = hp.read_map(f.replace("wmap", "weights"), field=[0, 4, 8])
        h = hp.read_map(f.replace("wmap", "hits"))
        wmap += m
        weights += w
        hits += h
        atomic_map_weights.append(0.5 * np.sum(w[1:, :], axis=(0, 1)))

    # Cut pixels with nonpositive weights
    weights[weights <= 0] = np.inf
    signal = wmap / weights

    if return_map_weights:
        return signal, weights, hits, np.array(atomic_map_weights)
    return signal, weights, hits


def coadd_maps(map_list, pixelization="hp", sign_list=None,
               res=None, dec_cut=None, return_map_weights=True):
    """
    """
    if pixelization == "hp":
        return _coadd_maps_hp(map_list, sign_list=sign_list,
                              res=res, dec_cut=dec_cut,
                              return_map_weights=return_map_weights)
    elif pixelization == "car":
        return _coadd_maps_car(map_list, sign_list=sign_list,
                               res=res, dec_cut=dec_cut,
                               return_map_weights=return_map_weights)


def main(args):
    """
    Read list of inverse noise-weighted atomic maps and list of per-pixel
    inverse noise covariance matrices, separate them into bundles, coadd
    them, adn write them to disk. Optionally apply a random sign flip to the
    atomics within each bundle before coadding, to obtain per-bundle noise
    realizations.

    Parameters
    ----------
        atomic_maps_list: list
            List of strings, containing the file paths of atomic maps.
        nbundles: integer
            Number of bundles to be formed from atomics.
        pixelization: str
            Type of pixelization ("hp" or "car").
        seed: int
            Random seed needed for shuffling the atomic maps before bundling.
        outdir: str
            Output directory.
        do_signflip: boolean, whether to apply the sign flip procedure.
                     Default: False.
        atomic_maps_weights: list
            List of float values corresponding to the overall map weight,
            needed for making sign-flip noise realizations. Default: None.
    """
    atomic_maps_list = np.load(args.atomic_maps_list)["wmap"]
    natomics = len(atomic_maps_list)
    nbundles = int(args.nbundles)

    # Random shuffle atomics list
    random.seed(args.seed)
    random.shuffle(atomic_maps_list)

    if args.do_signflip and args.atomic_maps_weights is None:
        raise ValueError("For sign-flip noise, atomic_maps_weights "
                         "must be provided.")

    os.makedirs(args.outdir, exist_ok=True)

    # Random division into bundles (using Yuji's code)
    bundle_mask = coordinator.gen_masks_of_given_atomic_map_list_for_bundles(
        natomics, nbundles
    )

    mpi.init(True)
    for id_bundle in mpi.taskrange(nbundles - 1):
        atom_msk = np.array(bundle_mask[id_bundle])
        atom_idx = np.arange(natomics)[atom_msk]
        fname_wmap_list = atomic_maps_list[atom_idx]
        natom = len(fname_wmap_list)
        print(f"Bundle # {id_bundle}: {natom} atomics")

        # Optionally apply signflip using Yuji's code
        if args.do_signflip:
            map_type = "signflip"
            obs_weights_fname = args.atomic_maps_weights.format(
                id_bundle=id_bundle
            )
            obs_weights = np.load(obs_weights_fname)["atomic_maps_weights"]
            assert len(obs_weights) == natom

            sf = coadder.SignFlip()
            sf.gen_seq(obs_weights)
            signs = sf.seq * 2 - 1
        else:
            map_type = "map"
            signs = np.ones_like(atom_idx)

        if args.pixelization == "car":
            car_kwargs = {"res": 10, "dec_cut": (-75, 25)}
        elif args.pixelization == "hp":
            car_kwargs = {"res": None, "dec_cut": None}
        else:
            raise ValueError("Not a supported pixelization type.")

        signal, weights, hits, obs_weights = coadd_maps(
            fname_wmap_list, pixelization=args.pixelization, sign_list=signs,
            return_map_weights=True, **car_kwargs
        )

        # Save to disk
        # This is just a proxy to the obs ID, e.g.
        # "atomic_1709852088_ws2_f090_full"
        obsid_list = [
            "_".join(atm.split('/')[-1].split("_")[:-1])
            for atm in atomic_maps_list[:natomics][atom_msk]
        ]
        np.savez(f"{args.outdir}/bundle{id_bundle}_atomics.npz",
                 atomic_maps_weights=obs_weights, obsid_list=obsid_list
                 )
        if args.pixelization == "car":
            enmap.write_map(f"{args.outdir}/bundle{id_bundle}_{map_type}.fits",
                            signal)
            enmap.write_map(f"{args.outdir}/bundle{id_bundle}_hits.fits", hits)
            enmap.write_map(f"{args.outdir}/bundle{id_bundle}_weights.fits",
                            weights)
        elif args.pixelization == "hp":
            hp.write_map(f"{args.outdir}/bundle{id_bundle}_{map_type}.fits.gz",
                         signal, overwrite=True, dtype=np.float32)
            hp.write_map(f"{args.outdir}/bundle{id_bundle}_hits.fits.gz", hits,
                         overwrite=True, dtype=np.float32)
            hp.write_map(f"{args.outdir}/bundle{id_bundle}_weights.fits.gz",
                         weights, overwrite=True, dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic_maps_list",
                        help="Path to npz file with path to atomic maps.")
    parser.add_argument("--pixelization",
                        help="Pixelization ('hp' or 'car')")
    parser.add_argument("--seed",
                        help="Random seed to reproduce shuffling of atomics.")
    parser.add_argument("--outdir",
                        help="Output directory for bundle maps list.")
    parser.add_argument("--nbundles",
                        help="Number of bundles to make from atomic maps.")
    parser.add_argument("--do_signflip", action="store_true",
                        help="Whether to make sign-flip noise realizations"
                        "from the atomic maps in each bundle.")
    parser.add_argument("--atomic_maps_weights",
                        help="Path to npz file with per-atomic weights needed"
                        "for sign-flip noise.")

    args = parser.parse_args()
    main(args)
