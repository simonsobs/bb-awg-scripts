import numpy as np
import healpy as hp
import toast
import os


def get_filtered_sim(mp, obsmat):
    """
    Apply an observation matrix to a QU polarization map, and return the
    filtered QU map.

    Parameters
    ---------
        mp: np.array
            TQU map to be filtered.
        obsmat: toast.Obsmat object
            Observation matrix to filter simulations.

    Returns
    -------
        mp_filtered: np.array
            Filtered TQU map.

    """
    mpT, mpQ, mpU = mp.copy()
    mpQ = hp.reorder(mpQ, r2n=True)
    mpU = hp.reorder(mpU, r2n=True)
    mpTf, mpQf, mpUf = obsmat.apply(np.hstack([mpT, mpQ, mpU])).reshape([3, -1])  # noqa: E501
    mpTf = hp.reorder(mpTf, n2r=True)
    mpQf = hp.reorder(mpQf, n2r=True)
    mpUf = hp.reorder(mpUf, n2r=True)

    return np.array([mpTf, mpQf, mpUf])


nside = 128
overwrite = True
filter_setup = f"obsmat_apo_nside{nside}"  # f"obsmat_polyonly_apo_nside{nside}"  # noqa: E501
what_sims = "pure"  # "cmb"
sim_ids = list(range(0, 1000))

if what_sims == "cmb":
    output_dir = f"/pscratch/sd/k/kwolz/bbdev/simpure/filtered_cmb_sims/{filter_setup}"  # noqa: E501
    sim_dir = "/pscratch/sd/k/kwolz/bbdev/simpure/cmb_sims"
    sim_types = ["cmbB", "cmbEB"]
elif what_sims == "pure":
    output_dir = f"/pscratch/sd/k/kwolz/bbdev/simpure/filtered_pure_sims/{filter_setup}"  # noqa: E501
    sim_dir = "/pscratch/sd/k/kwolz/bbdev/simpure/input_sims"
    sim_types = [f"pure{p}" for p in "E"]

os.makedirs(output_dir, exist_ok=True)
print(output_dir)
sim_string_format = "{sim_type}_nside128_fwhm30.0_sim{sim_id:04d}_HP.fits"
obsmat_dir = "/pscratch/sd/c/chervias/SimonsObs/BBMASTER/toast/output/obs_mat_nside128_fpthin8/obsmat_coadd-full.npz"  # "/global/cfs/cdirs/sobs/awg_bb/bbmaster_paper/obs_mat_nside128_fpthin8_onlypoly/obsmat_coadd-full.npz"  # noqa: E501
obsmat = toast.ObsMat(obsmat_dir)

for sim_id in sim_ids:
    if sim_id % 10 == 0:
        print(sim_id)
    for sim_type in sim_types:
        sim_fname = (
            sim_dir + "/" +
            sim_string_format.format(sim_type=sim_type, sim_id=sim_id)
        )
        out_fname = (
            output_dir + "/" +
            sim_string_format.format(sim_type=sim_type, sim_id=sim_id)
        )
        if os.path.isfile(out_fname) and not overwrite:
            continue
        mp = hp.read_map(sim_fname, field=range(3))
        mpf = get_filtered_sim(mp, obsmat)
        hp.write_map(out_fname, mpf, dtype=float, overwrite=overwrite)
        print(out_fname)
