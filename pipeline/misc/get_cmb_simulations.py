import healpy as hp
from pixell import enmap, enplot, curvedsky, uharm
import numpy as np
import os
import camb


def get_theory_cls(cosmo_params, lmax, lmin=0):
    """
    """
    params = camb.set_params(**cosmo_params)
    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit='K', raw_cl=True)
    lth = np.arange(lmin, lmax+1)

    cl_th = {
        "TT": powers["total"][:, 0][lmin:lmax+1],
        "EE": powers["total"][:, 1][lmin:lmax+1],
        "TE": powers["total"][:, 3][lmin:lmax+1],
        "BB": powers["total"][:, 2][lmin:lmax+1]
    }
    for spec in ["EB", "TB"]:
        cl_th[spec] = np.zeros_like(lth)

    return lth, cl_th


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
    smooth_fwhm = args.smooth_fwhm
    pix_type = args.pix_type
    pols_keep = args.pols_keep
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
    lmax_sim = lmax + 500

    cosmo = {
       "cosmomc_theta": 0.0104085,
       "As": 2.1e-9,
       "ombh2": 0.02237,
       "omch2": 0.1200,
       "ns": 0.9649,
       "Alens": 1.0,
       "tau": 0.0544,
       "r": 0.0,
    }

    lth, clth = get_theory_cls(
        cosmo,
        lmax=lmax_sim
    )
    # np.savez("clth_r0_AL1.npz", l=lth, **clth)

    pairs_keep = {
        "TEB": ["TT", "TE", "EE", "BB"],
        "EB": ["EE", "BB"],
        "E": ["EE"],
        "B": ["BB"],
        "T": ["TT"],
        "TE": ["TT", "TE"],
        "TB": ["TT", "BB"],
    }
    for fp in clth:
        if fp not in pairs_keep[pols_keep]:
            clth[fp] = np.zeros_like(clth[fp])

    if do_plots:
        os.makedirs(f"{out_dir}/plots", exist_ok=True)

    for id_sim in range(id_start, id_start + n_sims):
        print(f"sim {id_sim+1} / {n_sims}")
        alms = hp.synalm(
            [clth["TT"], clth["TE"], clth["EE"], clth["BB"]],
            lmax=lmax_sim
        )
        alms = hp.smoothalm(alms, fwhm=np.deg2rad(smooth_fwhm/60))
        if args.pix_type == "hp":
            map = hp.alm2map(alms, nside=nside)
            hp.write_map(
                f"{out_dir}/cmb{pols_keep}_nside{nside}_fwhm{smooth_fwhm}_sim{id_sim:04d}.fits",  # noqa
                map,
                overwrite=True
            )
        elif args.pix_type == "car":
            map = curvedsky.alm2map(alms, template, copy=True)
            enmap.write_map(
                f"{out_dir}/cmb{pols_keep}_{round(res)}arcmin_fwhm{smooth_fwhm}_sim{id_sim:04d}_CAR.fits",  # noqa
                map
            )
            print(f"{out_dir}/cmb{pols_keep}_{round(res)}arcmin_fwhm{smooth_fwhm}_sim{id_sim:04d}_CAR.fits")
            if do_plots:
                for i, fp in enumerate("TQU"):
                    lim = 3e-6 if i else 2e-4
                    plot = enplot.plot(
                        map[i], color="planck", ticks=10, range=lim,
                        colorbar=True
                    )
                    enplot.write(
                        f"{out_dir}/plots/cmb{pols_keep}_{int(res)}arcmin_fwhm{smooth_fwhm}_sim{id_sim:04d}_CAR_{fp}",  # noqa
                        plot
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
        "--smooth_fwhm",
        type=float, default=30,
        help="Smooth scale FWHM in arcmin"
    )
    parser.add_argument(
        "--n_sims",
        type=int,
        help="Number of simulations"
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
        "--pols_keep",
        help="Polarization types to keep, e.g. 'TEB', 'EB', 'B'. "
             "Others will be set to zero in the maps.",
        default="TEB"
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Do not make plots"
    )

    args = parser.parse_args()

    main(args)
