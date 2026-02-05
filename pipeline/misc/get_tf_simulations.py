import healpy as hp
import numpy as np
from pixell import enmap
from pixell import uharm
from pixell import curvedsky
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


def read_map(map_file,
             pix_type='hp',
             fields_hp=None,
             convert_K_to_muK=False,
             geometry=None):
    """
    ** From SOOPERCOOL/soopercool/map_utils.py **
    Read a map from a file, regardless of the pixellization type.

    Parameters
    ----------
    map_file : str
        Map file name.
    pix_type : str, optional
        Pixellization type.
    fields_hp : tuple, optional
        Fields to read from a HEALPix map.
    convert_K_to_muK : bool, optional
        Convert K to muK.
    geometry : enmap.geometry, optional
        Enmap geometry.

    Returns
    -------
    map_out : np.ndarray
        Loaded map.
    """
    conv = 1
    if convert_K_to_muK:
        conv = 1.e6
    _check_pix_type(pix_type)
    if pix_type == 'hp':
        kwargs = {"field": fields_hp} if fields_hp is not None else {}
        m = hp.read_map(map_file, **kwargs)
    else:
        m = enmap.read_map(map_file, geometry=geometry)

    return conv*m


def bandlim_sine2(x, xc, dx):
    xmin = xc - dx
    xmax = xc + dx
    return 1 - np.where(
        x < xmin,
        0,
        np.where(
            x > xmax,
            1.,
            np.sin(np.pi/2*(x-xmin)/(xmax-xmin))**2
        )
    )


def main(args):
    """
    """
    pix_type = args.pix_type
    n_sims = args.n_sims
    id_start = args.sim_id_start
    smooth_fwhm = args.smooth_fwhm
    pols_keep = args.pols_keep
    nside = args.nside
    car_template = args.car_template_map

    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        raise ValueError(f"Directory does not exist: {out_dir}")

    if pix_type == "car":
        if car_template is not None:
            print(f"Reading car_template_map: {car_template}")
            geometry = enmap.read_map_geometry(car_template)
            wcs = geometry[1]
            res_arcmin = np.min(np.abs(wcs.wcs.cdelt))*60.
        elif args.res_arcmin is not None:
            print(f"Using res_arcmin = {args.res_arcmin}")
            res_arcmin = args.res_arcmin
            geometry = get_fullsky_geometry(res_arcmin)
        else:
            raise ValueError("Either car_template_map or res_arcmin must "
                             "be given.")
        shape, wcs = geometry
        new_shape = (3,) + shape[-2:]
        template = enmap.zeros(new_shape, wcs)
        lmax = lmax_from_map(template, pix_type="car")
    else:
        if nside is None:
            raise ValueError("nside must be given.")

        lmax = 3 * nside - 1

    lmax_sim = lmax + 500
    ells = np.arange(lmax_sim + 1)
    beam = np.exp(-0.5*ells*(ells+1)*np.radians(smooth_fwhm/60.)**2)

    ps = 1 / (ells + 0.01)**2 * beam**2
    fl = bandlim_sine2(ells, 650, 50)

    pure_types = [f"pure{p}" for p in pols_keep]

    for id_sim in range(id_start, id_start+n_sims):
        print(f"sim {id_sim+1} / {id_start + n_sims}")
        np.random.seed(id_sim)
        alms = hp.synalm(ps, lmax=lmax_sim)
        alms = hp.almxfl(alms, fl)

        for tag in pure_types:
            i = {"pureT": 0, "pureE": 1, "pureB": 2}[tag]
            alms_list = 3*[alms*0]
            alms_list[i] = alms

            if pix_type == "hp":
                map = hp.alm2map(
                    alms_list, nside, lmax=lmax_sim
                )
                hp.write_map(
                    f"{out_dir}/{tag}_nside{nside}_fwhm{smooth_fwhm}_sim{id_sim:04d}_HP.fits",  # noqa
                    map,
                    overwrite=True,
                    dtype=float
                )
            elif pix_type == "car":
                map = curvedsky.alm2map(
                    alms_list,
                    template
                )
                enmap.write_map(
                    f"{out_dir}/{tag}_{res_arcmin:.1f}arcmin_fwhm{smooth_fwhm}_sim{id_sim:04d}_CAR.fits",  # noqa
                    map
                )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pix_type",
        help="Pixelization type, either 'hp' or 'car'."
    )
    parser.add_argument(
        "--car_template_map",
        help="Example map that has the same format and resolution as the sims",
        default=None
    )
    parser.add_argument(
        "--nside",
        type=int,
        help="Healpix nside",
        default=None
    )
    parser.add_argument(
        "--smooth_fwhm",
        type=float,
        help="Smoothing scale FWHM in arcmin",
        default=30
    )
    parser.add_argument(
        "--n_sims",
        type=int,
        help="Number of simulations"
    )
    parser.add_argument(
        "--sim_id_start",
        type=int,
        default=0,
        help="Simulation ID to start with"
    )
    parser.add_argument(
        "--out_dir",
        help="Output directory"
    )
    parser.add_argument(
        "--res_arcmin",
        type=float,
        help="Resolution in arcmin. "
             "Will be ignored if car_template_map is given."
    )
    parser.add_argument(
        "--pols_keep",
        help="Pure polarization types to keep, e.g. 'TEB' means that pureT,"
             "pureE, pureB will all be simulated.",
        default="TEB"
    )
    args = parser.parse_args()

    main(args)
