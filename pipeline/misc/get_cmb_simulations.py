import healpy as hp
import numpy as np
import os
from pixell import enmap
from pixell import uharm
from pixell import curvedsky
from pixell import enplot
import camb


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


def main(args):
    """
    """
    pix_type = args.pix_type
    n_sims = args.n_sims
    id_start = args.sim_id_start
    smooth_fwhm = args.smooth_fwhm
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

    _, clth = get_theory_cls(
        cosmo,
        lmax=lmax
    )

    for id_sim in range(id_start, id_start+n_sims):
        print(f"{id_sim+1}/{n_sims}")

        np.random.seed(id_sim)
        alms_list = hp.synalm([clth["TT"], clth["TE"], clth["EE"], clth["BB"]],
                              lmax=lmax)

        if pix_type == "hp":
            map = hp.alm2map(
                alms_list, nside=nside, lmax=lmax,
                fwhm=np.deg2rad(smooth_fwhm/60)
            )
            hp.write_map(
                f"{out_dir}/cmb_nside{nside}_fwhm{smooth_fwhm}_sim{id_sim:04d}.fits",  # noqa
                map,
                overwrite=True
            )
        elif pix_type == "car":
            map = curvedsky.alm2map(
                alms_list,
                template
            )
            enmap.write_map(
                f"{out_dir}/cmb_{res_arcmin:.1f}arcmin_fwhm{smooth_fwhm}_sim{id_sim:04d}_CAR.fits",  # noqa
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
    args = parser.parse_args()

    main(args)
