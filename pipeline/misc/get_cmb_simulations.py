import healpy as hp
import numpy as np
import os
from soopercool import utils

def main(args):
    """
    """
    nside = args.nside
    n_sims = args.n_sims
    smooth_fwhm = args.smooth_fwhm

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

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
    
    lth, clth = utils.get_theory_cls(
        cosmo,
        lmax=lmax
    )

    for id_sim in range(n_sims):
        map = hp.synfast([clth["TT"], clth["TE"], clth["EE"], clth["BB"]], lmax=lmax, nside=nside, fwhm=np.deg2rad(smooth_fwhm/60))

        hp.write_map(
            f"{out_dir}/cmb_nside{nside}_fwhm{smooth_fwhm}_sim{id_sim:04d}.fits",
            map,
            overwrite=True
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", type=int, help="Healpix nside")
    parser.add_argument("--smooth_fwhm", type=float, help="Smooth scale FWHM in arcmin")
    parser.add_argument("--n_sims", type=int, help="Number of simulations")
    parser.add_argument("--out_dir", type=str, help="Output directory")

    args = parser.parse_args()

    main(args)