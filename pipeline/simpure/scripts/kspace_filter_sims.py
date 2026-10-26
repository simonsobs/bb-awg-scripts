import numpy as np
from pixell import enmap
import os
import sys
sys.path.insert(0, "/global/homes/k/kwolz/bbdev/bb-awg-scripts/pipeline/misc")
import mpi_utils as mpi  # noqa
import utils as ut  # noqa


def binarize_mask(mask, pix_type="hp", threshold=0.):
    """
    From https://github.com/simonsobs/SOOPERCOOL/blob/tf-validation/soopercool/map_utils.py#L746  # noqa: E501

    Make a binary mask out of a nonbinary mask,
    including optional thresholding at > 0.

    Parameters
    ----------
    mask: np.ndarray or enmap.ndmap
        Input mask.
    pix_type : str, optional
        Pixelization type.
    threshold: float
        Threshold below which pixels in th enonbinary mask are cut.

    Returns
    -------
    mask_out: np.ndarray or enmap.ndmap
        Output mask.
    """
    mask_out = (mask > threshold).astype(np.float64)
    if pix_type == "hp":
        return mask_out
    else:
        return enmap.ndmap(mask_out, wcs=mask.wcs)


arcmin = 20
beam_fwhm = 60
ignore_T = True  # Do not filter T
overwrite = True
base_dir = "/pscratch/sd/k/kwolz/bbdev/simpure"  # NERSC
filter_setup = "kx40"
what_sims = "plaw"  # "cmb"  # "pure" # "plaw" # "noise"
sim_ids = list(range(0, 200))

kspace_pars = {
    "kx20": {"dkx": 20., "dky": 0., "type": "cosine"},
    "kx40": {"dkx": 40., "dky": 0., "type": "cosine"}
}

# MPI initialization
rank, size, comm = mpi.init(True)

output_dir = f"{base_dir}/filtered_{what_sims}_sims/{filter_setup}"
if what_sims == "cmb":
    sim_dir = f"{base_dir}/cmb_sims"
    sim_types = ["cmbB", "cmbEB"]
    sim_string_format = "{sim_type}_"+str(arcmin)+f"arcmin_fwhm{int(beam_fwhm)}"+"_sim{sim_id:04d}_CAR.fits"  # noqa: E501
elif what_sims == "pure":
    sim_dir = f"{base_dir}/input_sims"
    sim_types = [f"pure{p}" for p in "TB"]
    sim_string_format = "{sim_type}_"+str(arcmin)+f"arcmin_fwhm{int(beam_fwhm)}"+"_sim{sim_id:04d}_CAR.fits"  # noqa: E501
elif what_sims == "plaw":
    sim_dir = f"{base_dir}/plaw_sims"
    sim_types = ["plawB", "plawEB"]
    sim_string_format = "{sim_type}_"+str(arcmin)+f"arcmin_fwhm{int(beam_fwhm)}"+"_sim{sim_id:04d}_CAR.fits"  # noqa: E501
elif what_sims == "noise":
    sim_dir = f"{base_dir}/noise_sims"
    sim_types = ["white_noise_7muKamin"]
    sim_string_format = "{sim_type}_"+str(arcmin)+"arcmin_sim{sim_id:04d}__{bundle_id}_CAR.fits"  # noqa: E501
else:
    raise ValueError("This type of sims is unknown.")

os.makedirs(output_dir, exist_ok=True)
print(output_dir)

mask_file = "/global/homes/k/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/data/mask_apo_fullsky_"+str(arcmin)+"arcmin.fits"  # NERSC  # noqa: E501
mask = enmap.read_map(mask_file)
mask_bin = binarize_mask(mask)
bundle_ids = [0, 1] if what_sims == "noise" else [0]

# Every rank must have the same shared list
mpi_shared_list = [(sid, bid) for sid in sim_ids for bid in bundle_ids]
mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list),
                                logger=None)
local_mpi_list = [mpi_shared_list[i] for i in task_ids]

for sim_id, bundle_id in local_mpi_list:
    if sim_id % 10 == 0 and bundle_id == 0:
        print(sim_id)
    for sim_type in sim_types:
        print(sim_string_format)
        if what_sims == "noise":
            sim_string = sim_string_format.format(sim_type=sim_type,
                                                  sim_id=sim_id,
                                                  bundle_id=bundle_id)
        else:
            sim_string = sim_string_format.format(sim_type=sim_type,
                                                  sim_id=sim_id)
        sim_fname = sim_dir + "/" + sim_string
        out_fname = output_dir + "/" + sim_string
        if what_sims == "noise":
            sim_fname = sim_fname.format(bundle_id=bundle_id)
            out_fname = out_fname.format(bundle_id=bundle_id)
        if os.path.isfile(out_fname) and not overwrite:
            continue
        mp = enmap.read_map(sim_fname)
        mpf = ut.kspace_filter_sim(
            mp,
            dkx=kspace_pars[filter_setup]["dkx"],
            dky=kspace_pars[filter_setup]["dky"],
            type=kspace_pars[filter_setup]["type"])
        enmap.write_map(out_fname, mpf)
        print(out_fname)
comm.barrier()
