import healpy as hp
import numpy as np
import os

def main(args):
    """
    """
    nside = args.nside
    n_sims = args.n_sims
    smooth_fwhm = args.smooth_fwhm

    out_dir = "tf_sims"
    os.makedirs(out_dir, exist_ok=True)

    lmax = 3 * nside - 1
    ells = np.arange(lmax + 1)

    ps = 1 / (ells + 10) ** 2

    for id_sim in range(n_sims):
        alms = hp.synalm(ps, lmax=lmax)

        for i, tag in enumerate(["pureT", "pureE", "pureB"]):
            
            alms_list = np.zeros((3, *alms.shape), dtype=np.complex128)
            alms_list[i, :] += alms
            map = hp.alm2map(alms_list, nside=nside, lmax=lmax, fwhm=np.deg2rad(smooth_fwhm/60))

            hp.write_map(
                f"{out_dir}/{tag}_nside{nside}_fwhm{smooth_fwhm}_sim{id_sim:04d}.fits",
                map,
                overwrite=True
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", type=int, help="Healpix nside")
    parser.add_argument("--smooth_fwhm", type=float, help="Smooth scale FWHM in arcmin")
    parser.add_argument("--n_sims", type=int, help="Number of simulations")

    args = parser.parse_args()

    main(args)