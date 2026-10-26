import healpy as hp
from pixell import enmap, curvedsky, uharm
# from pixell import enplot
import numpy as np
import os


def _check_pix_type(pix_type):
    """
    ** From SOOPERCOOL/soopercool/map_utils.py **
    Error handling for pixellization types.

    Parameters
    ----------
    pix_type : str
        Pixellization type.
    """
    if not (pix_type in ['hp', 'car']):
        raise ValueError(f"Unknown pixelisation type {pix_type}.")


def get_fullsky_geometry(res_arcmin=1., variant="fejer1"):
    """
    Generates a fullsky CAR template at resolution res-arcmin.
    """
    res = res_arcmin * np.pi/180/60
    return enmap.fullsky_geometry(res=res, proj='car', variant=variant)


def lmax_from_map(map, pix_type="hp"):
    """
    ** From SOOPERCOOL/soopercool/map_utils.py **
    Determine the maximum multipole from a map and its
    pixellization type.

    Parameters
    ----------
    map : np.ndarray or enmap.ndmap
        Input map.
    pix_type : str, optional
        Pixellization type.

    Returns
    -------
    int
        Maximum multipole.
    """
    _check_pix_type(pix_type)
    if pix_type == "hp":
        nside = hp.npix2nside(map.shape[-1])
        return 3 * nside - 1
    else:
        _, wcs = map.geometry
        res = np.deg2rad(np.min(np.abs(wcs.wcs.cdelt)))
        lmax = uharm.res2lmax(res)
        return lmax


def main(args):
    """
    """
    nside = args.nside
    n_sims = args.n_sims
    id_start = args.id_start
    n_bundles = args.n_bundles
    pix_type = args.pix_type
    noise_muK_amin = args.noise_muK_amin
    do_plots = not args.no_plots

    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        raise ValueError(f"Directory does not exist: {out_dir}")

    if pix_type == "car":
        if args.car_template is not None:
            geometry = enmap.read_map_geometry(args.car_template)
            wcs = geometry[1]
            res = np.min(np.abs(wcs.wcs.cdelt)) * 60.
        elif args.res_arcmin is not None:
            geometry = get_fullsky_geometry(args.res_arcmin)
            res = args.res_arcmin
        else:
            raise ValueError("car_template of res_arcmin required.")
        shape, wcs = geometry
        new_shape = (3,) + shape[-2:]
        template = enmap.zeros(new_shape, wcs)
        lmax = lmax_from_map(template, pix_type="car")
    else:
        lmax = 3 * nside - 1
    lmax_sims = lmax + 500

    if do_plots:
        os.makedirs(f"{out_dir}/plots", exist_ok=True)

    for id_sim in range(id_start, id_start + n_sims):
        print(f"sim {id_sim+1} / {id_start + n_sims}")
        for id_bundle in range(n_bundles):
            # TODO: we'd like to add 1/f noise, or filter it. Currently we
            # ignore filtering noise.
            noise_theory_spectrum = (noise_muK_amin * np.pi / (180.0 * 60.0))**2  # noqa: E501
            Nell = np.array(
                [fac*noise_theory_spectrum*np.ones(lmax_sims+1)
                 for fac in [0.5, 0.5, 1., 1.]],  # TT, TE, EE, BB noise
                dtype=np.float64
            )
            np.random.seed(n_bundles*(n_sims+id_sim)+id_bundle)

            alms = hp.synalm(Nell)

            if args.pix_type == "hp":
                map = hp.alm2map(alms, nside=nside)
                hp.write_map(
                    f"{out_dir}/white_noise_{round(noise_muK_amin)}muKamin_nside{nside}_sim{id_sim:04d}__{id_bundle}_HP.fits",  # noqa
                    map,
                    overwrite=True,
                    dtype=np.float64
                )
            elif args.pix_type == "car":
                map = curvedsky.alm2map(alms, template, copy=True)
                enmap.write_map(
                    f"{out_dir}/white_noise_{round(noise_muK_amin)}muKamin_{round(res)}arcmin_sim{id_sim:04d}__{id_bundle}_CAR.fits",  # noqa
                    map
                )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pix_type",
        help="Pixel type; hp or car"
    )
    parser.add_argument(
        "--car_template",
        help="CAR geometry template. If None, assume fullsky geometry.",
        default=None
    )
    parser.add_argument(
        "--res_arcmin",
        type=float,
        help="CAR geometry resolution in arcmin. Ignored if car_template.",
        default=None
    )
    parser.add_argument(
        "--nside",
        type=int,
        help="Healpix nside"
    )
    parser.add_argument(
        "--n_bundles",
        type=int,
        help="Number of noise bundles to cross-correlate"
    )
    parser.add_argument(
        "--n_sims",
        type=int,
        help="Number of simulations"
    )
    parser.add_argument(
        "--noise_muK_amin",
        type=float,
        help="White-noise level in muK*arcmin"
    )
    parser.add_argument(
        "--id_start",
        type=int,
        default=0,
        help="Start ID of simulations"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory"
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Do not make plots"
    )

    args = parser.parse_args()

    main(args)
