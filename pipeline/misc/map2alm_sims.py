import healpy as hp
import numpy as np
import os
import sys

sys.path.append("../misc")
import mpi_utils as mpi  # noqa


def main(args):
    """
    Convert HEALPix map simulations to alms compatible with SOOPERCOOL.
    """
    nside = args.nside
    n_sims = args.n_sims

    out_dir = args.out_dir
    map_fname_format = args.map_fname_format
    alm_fname_format = args. alm_fname_format
    os.makedirs(out_dir, exist_ok=True)

    lmax = 3 * nside - 1

    mpi.init(True)

    for sim_id in mpi.taskrange(n_sims - 1):
        print(f"  {sim_id:04}")
        map = hp.read_map(
            f"{out_dir}/{map_fname_format.format(sim_id=sim_id)}",
            field=range(3)
        )
        alm = hp.map2alm(map, lmax=lmax)
        hp.write_alm(
            f"{out_dir}/{alm_fname_format.format(sim_id=sim_id)}",
            alm, overwrite=True, out_dtype=np.float32
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", type=int, help="Healpix nside")
    parser.add_argument("--n_sims", type=int, help="Number of simulations")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    parser.add_argument(
        "--map_fname_format", type=str,
        help="Naming convention for input maps; must contain formatter sim_id"
    )
    parser.add_argument(
        "--alm_fname_format", type=str,
        help="Naming convention for output alms; must contain formatter sim_id"
    )

    args = parser.parse_args()

    main(args)
