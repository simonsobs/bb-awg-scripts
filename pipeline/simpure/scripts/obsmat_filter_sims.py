import numpy as np
import healpy as hp
import toast
import os


def get_filtered_sim(mp, obsmat, ignore_T=False, mask_bin=None):
    """
    Apply an observation matrix to a QU polarization map, and return the
    filtered QU map.

    Parameters
    ---------
        mp: np.array
            TQU map to be filtered.
        obsmat: toast.Obsmat object
            Observation matrix to filter simulations.
        mask_binary: np.array
            Binary mask to (optionally) mask the map before applying the obsmat

    Returns
    -------
        mp_filtered: np.array
            Filtered TQU map.

    """
    if mask_bin is None:
        mask_bin = np.ones_like(mp[0])
    mpT, mpQ, mpU = mask_bin[None, :] * mp.copy()
    mpQ = hp.reorder(mpQ, r2n=True)
    mpU = hp.reorder(mpU, r2n=True)
    mpTf, mpQf, mpUf = obsmat.apply(np.hstack([mpT, mpQ, mpU])).reshape([3, -1])  # noqa: E501
    mpTf = hp.reorder(mpTf, n2r=True)
    mpQf = hp.reorder(mpQf, n2r=True)
    mpUf = hp.reorder(mpUf, n2r=True)

    if ignore_T:
        np.array([mp.copy()[0], mpQf, mpUf])
    return np.array([mpTf, mpQf, mpUf])


nside = 64
beam_fwhm = 60
ignore_T = True  # Do not filter T
overwrite = True
base_dir = "/pscratch/sd/k/kwolz/bbdev/simpure"  # NERSC
# filter_setup = f"obsmat_apo_nside{nside}"  # BBMASTER paper
filter_setup = f"obsmat_polyonly_apo_nside{nside}"  # Simple polynomial filter
what_sims = "cmb"  # "cmb"  # "pure" # "plaw"
sim_ids = list(range(0, 200))

output_dir = f"{base_dir}/filtered_{what_sims}_sims/{filter_setup}_binary_masked"
if what_sims == "cmb":
    sim_dir = f"{base_dir}/cmb_sims"
    sim_types = ["cmbB", "cmbEB"]
elif what_sims == "pure":
    sim_dir = f"{base_dir}/input_sims"
    sim_types = [f"pure{p}" for p in "E"]
elif what_sims == "plaw":
    sim_dir = f"{base_dir}/plaw_sims"
    sim_types = ["plawB", "plawEB"]
else:
    raise ValueError("This type of sims is unknown.")

os.makedirs(output_dir, exist_ok=True)
print(output_dir)
sim_string_format = "{sim_type}_nside"+str(nside)+f"_fwhm{beam_fwhm:.1f}"+"_sim{sim_id:04d}_HP.fits"  # noqa: E501
if filter_setup == f"obsmat_polyonly_apo_nside{nside}":
    obsmat_dir = f"/global/cfs/cdirs/sobs/awg_bb/bbmaster_paper/obs_mat_nside{nside}_fpthin8_onlypoly/obsmat_coadd-full.npz"  # noqa: E501
else:
    obsmat_dir = f"/pscratch/sd/c/chervias/SimonsObs/BBMASTER/toast/output/obs_mat_nside{nside}_fpthin8/obsmat_coadd-full.npz"  # noqa: E501
obsmat = toast.ObsMat(obsmat_dir)
mask_file = f"/global/homes/k/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/data/mask_apo_nside{nside}.fits"  # NERSC # noqa: E501
# mask_file = f"/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/data/mask_apo_nside{nside}.fits"  # SO:UK noqa: E501
mask_bin = hp.read_map(mask_file, dtype=bool)


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
        mpf = get_filtered_sim(mp, obsmat, ignore_T, mask_bin)
        hp.write_map(out_fname, mpf, dtype=float, overwrite=overwrite)
        print(out_fname)
