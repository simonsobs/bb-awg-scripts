import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinvh
import pymaster as nmt
import os
import healpy as hp
from pixell import enmap
import camb

from soopercool import ps_utils as pu
from soopercool import map_utils as mu
from soopercool import coupling_utils as cu
from soopercool import utils as ut


def get_theory_cls(cosmo_params, lmax, lmin=0, beam_fwhm=None):
    """
    """
    params = camb.set_params(**cosmo_params)
    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=True)
    lth = np.arange(lmin, lmax+1)

    cl_th = {
        "TT": powers["total"][:, 0][lmin:lmax+1],
        "EE": powers["total"][:, 1][lmin:lmax+1],
        "TE": powers["total"][:, 3][lmin:lmax+1],
        "BB": powers["total"][:, 2][lmin:lmax+1]
    }
    for spec in ["EB", "TB"]:
        cl_th[spec] = np.zeros_like(lth)

    if beam_fwhm is not None:
        bl = hp.gauss_beam(np.radians(beam_fwhm/60.), lmax=lmax)
        cl_th = {spec: cl_th[spec]*bl**2 for spec in cl_th}

    return lth, cl_th


def get_fullsky_geometry(res_arcmin=1., variant="fejer1"):
    """
    Generates a fullsky CAR template at resolution res-arcmin.
    """
    res = res_arcmin * np.pi/180/60
    return enmap.fullsky_geometry(res=res, proj='car', variant=variant)


def get_inv_coupling(mask, nmt_bins, transfer=None, nmt_purify=False,
                     return_bp_win=False, wcs=None):
    """
    """
    tf_correct = transfer is not None
    pix_type = "hp" if wcs is None else "car"

    pure_label = "_nmt_purify" if nmt_purify else ""
    tf_label = "_tf_correct" if tf_correct else ""
    coupling_fname = f"{out_dir}/inv_coupling{tf_label}{pure_label}.npz"

    if os.path.isfile(coupling_fname):
        inv_coupling = np.load(coupling_fname)["inv_coupling"]
        if not return_bp_win:
            return inv_coupling
        else:
            try:
                bp_win = np.load(coupling_fname)["bp_win"]
                return inv_coupling, bp_win
            except KeyError:
                pass

    lmax = mu.lmax_from_map(mask, pix_type=pix_type)
    nl = lmax + 1
    nbins = len(nmt_bins.get_effective_ells())

    # NaMaster gives us a 7x7 MCM (in the field pair axes)
    binner = np.array([nmt_bins.bin_cell(np.array([cl]))[0]
                      for cl in np.eye(nl)]).T
    f_spin2 = nmt.NmtField(mask, None, spin=2, wcs=wcs, purify_b=nmt_purify)
    f_spin0 = nmt.NmtField(mask, None, spin=0, wcs=wcs)
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f_spin0, f_spin2, nmt_bins, is_teb=True)
    mcm = np.transpose(w.get_coupling_matrix().reshape([nl, 7, nl, 7]),
                       axes=[1, 0, 3, 2])

    # We copy over (TE, TB) to (ET, EB) in the new 9x9 MCM
    mcm_full = np.zeros((9, nl, 9, nl))
    mcm_full[:3, :, :3, :] = mcm[:3, :, :3, :]
    mcm_full[3:5, :, 3:5, :] = mcm[1:3, :, 1:3, :]
    mcm_full[5:9, :, 5:9, :] = mcm[3:7, :, 3:7, :]
    mcm = mcm_full

    # We (half-) bin the MCM
    bmcm = np.einsum('ij,kjlm->kilm', binner, mcm)
    bmcmb = np.transpose(
        np.array([
            np.sum(bmcm[:, :, :, nmt_bins.get_ell_list(i)],
                   axis=-1)
            for i in range(nbins)
        ]), axes=[1, 2, 3, 0]
    )

    if tf_correct:
        bmcm = np.einsum('ijk,jklm->iklm', transfer, bmcm)
        bmcmb = np.einsum('ijk,jklm->iklm', transfer, bmcmb)

    inv_coupling = np.linalg.inv(bmcmb.reshape((9*nbins, 9*nbins)))
    bp_win = np.dot(inv_coupling, bmcm.reshape([9*nbins, 9*nl])).reshape([9, nbins, 9, nl])  # noqa
    np.savez(coupling_fname, inv_coupling=inv_coupling, bp_win=bp_win)
    print(f"Saved coupling to {coupling_fname}")

    if return_bp_win:
        return inv_coupling, bp_win
    return inv_coupling


print("  0. Reading inputs")

# pixelization related
nside = None
pix_type = "car"
res_arcmin = None
# car_template = None
car_template = "/home/kw6905/bbdev/bb-awg-scripts/pipeline/simpure/band_car_fejer1_20arcmin.fits"  # noqa
beam_fwhm = 30

if pix_type == "car":
    if car_template is not None:
        geometry = enmap.read_map_geometry(car_template)
        wcs = geometry[1]
        res = np.min(np.abs(wcs.wcs.cdelt)) * 60.
    elif res_arcmin is not None:
        geometry = get_fullsky_geometry(res_arcmin)
        res = res_arcmin
    else:
        raise ValueError("car_template of res_arcmin required.")
    shape, wcs = geometry
    new_shape = (3,) + shape[-2:]
    template = enmap.zeros(new_shape, wcs)
    lmax = mu.lmax_from_map(template, pix_type="car")
else:
    lmax = 3 * nside - 1
    wcs = None
lmax_sim = lmax + 500


# binning
bin_label = "_all_bins"
#binning_file = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/soopercool_inputs/binning/binning_car_lmax540_deltal10_large_first_bin.npz"  # noqa
binning_file = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/soopercool_inputs/binning/binning_car_lmax540_deltal10.npz"  # noqa
binning = np.load(binning_file)
nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"], binning["bin_high"] + 1)
leff = nmt_bins.get_effective_ells()
print(leff)

# cosmo theory related
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
_, clth = get_theory_cls(cosmo, lmax=lmax_sim, beam_fwhm=beam_fwhm)
clbb_in = clth["BB"][:lmax+1]

# general
nsims_purify = 20
nsims_cmb = 10
nsims_transfer = 50
nsims_deproj = 50

apo_scale = 10
apo_type = "C1"
box_str = "_-47_-33_-10_130"
out_dir = f"/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/purification/butter4_cutoff_1e-2/apo{apo_scale}_{apo_type}{box_str}{bin_label}"  # noqa
plot_dir = f"{out_dir}/plots"
os.makedirs(plot_dir, exist_ok=True)
lmax_plot = 500

# apodized mask
mask_file = f"/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/soopercool_inputs/masks/analysis_mask_apo{apo_scale}_{apo_type}{box_str}.fits"  # noqa
mask = mu.read_map(mask_file,
                   pix_type=pix_type,
                   car_template=car_template)
# mask_bin = enmap.ndmap((mask > 0).astype(float), wcs=wcs)


def load_cmb_sim(sim_id, filtered=False, pols_keep="EB"):
    """
    """
    base_dir = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure"
    suffix = "_f090_science_filtered" if filtered else ""
    sim_dir = "satp3/south/f090/cmb_sims/butter4_cutoff_1e-2/coadded_sims"
    if not filtered:
        sim_dir = "cmb_sims"
    sim_fname = f"cmb{pols_keep}_20arcmin_fwhm30.0_sim{sim_id:04d}_CAR{suffix}.fits"  # noqa

    return mu.read_map(
        f"{base_dir}/{sim_dir}/{sim_fname}",
        pix_type=pix_type,
        fields_hp=[0, 1, 2],
        car_template=car_template,
        convert_K_to_muK=True
    )


def load_purification_sim(sim_id, filtered=False):
    """
    """
    base_dir = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure"
    suffix = "_f090_science_filtered" if filtered else ""
    sim_dir = "satp3/south/f090/transfer_function/butter4_cutoff_1e-2/coadded_sims"  # noqa
    if not filtered:
        sim_dir = "input_sims"
    sim_fname = f"pureE_20.0arcmin_fwhm30.0_sim{sim_id:04d}_CAR{suffix}.fits"

    return mu.read_map(
        f"{base_dir}/{sim_dir}/{sim_fname}",
        pix_type=pix_type,
        fields_hp=[0, 1, 2],
        car_template=car_template,
        convert_K_to_muK=True
    )


def load_transfer_sim(sim_id, filtered=False, type="pureT"):
    """
    """
    base_dir = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure"
    suffix = "_f090_science_filtered" if filtered else ""
    sim_dir = "satp3/south/f090/transfer_function/butter4_cutoff_1e-2/coadded_sims"  # noqa
    if not filtered:
        sim_dir = "input_sims"
    sim_fname = f"{type}_20.0arcmin_fwhm30.0_sim{sim_id:04d}_CAR{suffix}.fits"

    return mu.read_map(
        f"{base_dir}/{sim_dir}/{sim_fname}",
        pix_type=pix_type,
        fields_hp=[0, 1, 2],
        car_template=car_template,
        convert_K_to_muK=True
    )


def get_masked_map(mask, mp, nmt_purify=False):
    """
    """
    assert mp.shape[0] in [2, 3]

    if not nmt_purify:
        if pix_type == "hp":
            return mp*mask
        else:  # car
            return enmap.ndmap(mp*mask[None, :, :], wcs=wcs)
    else:
        f = nmt.NmtField(mask, mp[-2:], purify_b=True, wcs=wcs)
        mp_masked = np.array([m.reshape(shape) for m in f.get_maps()])
        mp_masked = np.flip(mp_masked, axis=(1, 2))  # weird CAR ordering
        if mp.shape[0] == 3:  # TQU maps
            mp_masked = np.array([mp[0]*mask, mp_masked[0], mp_masked[1]])
        if pix_type == "hp":
            return mp_masked
        else:  # car
            return enmap.ndmap(mp_masked, wcs=wcs)


# def compute_pspec_nmt(map, mask, tf_correct=False, nmt_purify=False, map2=None):
#     """
#     """
#     field_dict = {"TT": ("0x0", 0),
#                   "TE": ("0x2", 0),
#                   "TB": ("0x2", 1),
#                   "ET": ("2x0", 0),
#                   "BT": ("2x0", 1),
#                   "EE": ("2x2", 0),
#                   "EB": ("2x2", 1),
#                   "BE": ("2x2", 2),
#                   "BB": ("2x2", 3)}
#     fa_0 = nmt.NmtField(mask, [map[0]], wcs=wcs)
#     fa_2 = nmt.NmtField(mask, [map[1], map[2]], wcs=wcs, purify_b=nmt_purify)
#     if map2 is None:
#         (fb_0, fb_2) = (fa_0, fa_2)
#     else:
#         fb_0 = nmt.NmtField(mask, map2[0], wcs=wcs)
#         fb_2 = nmt.NmtField(mask, [map2[1], map[2]], wcs=wcs,
#                             purify_b=nmt_purify)
#     cl_dict = {
#         "0x0": nmt.compute_full_master(fa_0, fb_0, nmt_bins),
#         "0x2": nmt.compute_full_master(fa_0, fb_2, nmt_bins),
#         "2x0": nmt.compute_full_master(fa_2, fb_0, nmt_bins),
#         "2x2": nmt.compute_full_master(fa_2, fb_2, nmt_bins),
#     }
#     return {fp: cl_dict[idx[0]][idx[1]] for fp, idx in field_dict.items()}


def compute_pspec(map, mask, transfer=None, nmt_purify=False, map2=None,
                  wcs=None, return_just_fields=False):
    """
    """
    field = {
        "spin0": nmt.NmtField(mask, map[:1], wcs=wcs),
        "spin2": nmt.NmtField(mask, map[1:], wcs=wcs, purify_b=nmt_purify)
    }
    map2 = map
    field2 = field

    if map2 is not None:
        field2 = {
            "spin0": nmt.NmtField(mask, map2[:1], wcs=wcs),
            "spin2": nmt.NmtField(mask, map2[1:], wcs=wcs, purify_b=nmt_purify)
        }
    if return_just_fields:
        return field, field2

    pcls = pu.get_coupled_pseudo_cls(field, field2, nmt_bins)
    inv_coupling = get_inv_coupling(
        mask, nmt_bins, transfer=transfer, nmt_purify=nmt_purify, wcs=wcs
    )

    return pu.decouple_pseudo_cls(pcls, inv_coupling)


# Compute transfer function
print("  1. transfer function computation")

print("  1A. TF sims unfiltered w/o purification")
fname = out_dir + "/cls_tf_unfiltered_nopure_{id_sim:04d}.npz"
if os.path.isfile(fname):
    cls_tf_unfiltered_nopure = np.load(fname, allow_pickle=True)["cls"]
else:
    cls_tf_unfiltered_nopure = []
    for i in range(nsims_transfer):
        if i % 50 == 0:
            print("   ", i)
            fields, fields2 = ({}, {})
            for typ in [f"pure{f}" for f in "TEB"]:
                mp = load_transfer_sim(i, filtered=False, type=typ)
                f1, f2 = compute_pspec(mp, mask, nmt_purify=False, wcs=wcs,
                                       return_just_fields=True)
                fields[typ] = f1
                fields2[typ] = f2
            print(fields.keys(), fields2.keys())
            pcls_mat = pu.get_pcls_mat_transfer(fields, nmt_bins, fields2)
        cls_tf_unfiltered_nopure.append(pcls_mat)
    cls_tf_unfiltered_nopure = np.array(cls_tf_unfiltered_nopure)
    np.savez(fname, cls=cls_tf_unfiltered_nopure)

print("  1B. TF sims unfiltered w/ purification")
fname = out_dir + "/cls_tf_unfiltered_pure_{id_sim:04d}.npz"
if os.path.isfile(fname):
    cls_tf_unfiltered_pure = np.load(fname, allow_pickle=True)["cls"]
else:
    cls_tf_unfiltered_pure = []
    for i in range(nsims_transfer):
        if i % 50 == 0:
            print("   ", i)
            fields, fields2 = ({}, {})
            for typ in [f"pure{f}" for f in "TEB"]:
                mp = load_transfer_sim(i, filtered=False, type=typ)
                f1, f2 = compute_pspec(mp, mask, nmt_purify=True, wcs=wcs,
                                       return_just_fields=True)
                fields[typ] = f1
                fields2[typ] = f2
            print(fields.keys(), fields2.keys())
            pcls_mat = pu.get_pcls_mat_transfer(fields, nmt_bins, fields2)
        cls_tf_unfiltered_pure.append(pcls_mat)
    cls_tf_unfiltered_pure = np.array(cls_tf_unfiltered_pure)
    np.savez(fname, cls=cls_tf_unfiltered_pure)

print("  1C. TF sims filtered w/o purification")
fname = out_dir + "/cls_tf_filtered_nopure_{id_sim:04d}.npz"
if os.path.isfile(fname):
    cls_tf_filtered_nopure = np.load(fname, allow_pickle=True)["cls"]
else:
    cls_tf_filtered_nopure = []
    for i in range(nsims_transfer):
        if i % 50 == 0:
            print("   ", i)
            fields, fields2 = ({}, {})
            for typ in [f"pure{f}" for f in "TEB"]:
                mp = load_transfer_sim(i, filtered=True, type=typ)
                f1, f2 = compute_pspec(mp, mask, nmt_purify=False, wcs=wcs,
                                       return_just_fields=True)
                fields[typ] = f1
                fields2[typ] = f2
            pcls_mat = pu.get_pcls_mat_transfer(fields, nmt_bins, fields2)
        cls_tf_filtered_nopure.append(pcls_mat)
    cls_tf_filtered_nopure = np.array(cls_tf_filtered_nopure)
    np.savez(fname, cls=cls_tf_filtered_nopure)

print("  1D. TF sims filtered w/ purification")
fname = out_dir + "/cls_tf_filtered_pure_{id_sim:04d}.npz"
if os.path.isfile(fname):
    cls_tf_filtered_pure = np.load(fname, allow_pickle=True)["cls"]
else:
    cls_tf_filtered_pure = []
    for i in range(nsims_transfer):
        if i % 50 == 0:
            print("   ", i)
            fields, fields2 = ({}, {})
            for typ in [f"pure{f}" for f in "TEB"]:
                mp = load_transfer_sim(i, filtered=True, type=typ)
                f1, f2 = compute_pspec(mp, mask, nmt_purify=True, wcs=wcs,
                                       return_just_fields=True)
                fields[typ] = f1
                fields2[typ] = f2
            pcls_mat = pu.get_pcls_mat_transfer(fields, nmt_bins, fields2)
        cls_tf_filtered_pure.append(pcls_mat)
    cls_tf_filtered_pure = np.array(cls_tf_filtered_pure)
    np.savez(fname, cls=cls_tf_filtered_pure)

print("  1E. TF w/ purification")
fp_dum = ("None", "None")
pcls_mat_filt_dict = {fp_dum: {"filtered": cls_tf_filtered_pure}}
mean_pcls_mat_filt_dict = {fp_dum: np.mean(cls_tf_filtered_pure, axis=0)}
mean_pcls_mat_unfilt_dict = {fp_dum: np.mean(cls_tf_unfiltered_pure, axis=0)}
transfer_pure = cu.get_transfer_dict(
    mean_pcls_mat_filt_dict, mean_pcls_mat_unfilt_dict, pcls_mat_filt_dict,
    [fp_dum]
)[fp_dum]

field_pairs = ["EE", "EB", "BE", "BB"]
file_name = plot_dir + "/transfer_pure.pdf"
ut.plot_transfer_function(leff, transfer_pure, 0, lmax, field_pairs,
                          file_name)
plt.clf()
print(f"  PLOT SAVED {plot_dir}/transfer_pure.pdf")

print("  1F. TF w/o purification")
fp_dum = ("None", "None")
pcls_mat_filt_dict = {fp_dum: {"filtered": cls_tf_filtered_nopure}}
mean_pcls_mat_filt_dict = {fp_dum: np.mean(cls_tf_filtered_nopure, axis=0)}
mean_pcls_mat_unfilt_dict = {fp_dum: np.mean(cls_tf_unfiltered_nopure, axis=0)}
transfer_nopure = cu.get_transfer_dict(
    mean_pcls_mat_filt_dict, mean_pcls_mat_unfilt_dict, pcls_mat_filt_dict,
    [fp_dum]
)[fp_dum]

field_pairs = ["EE", "EB", "BE", "BB"]
file_name = plot_dir + "/transfer_nopure.pdf"
ut.plot_transfer_function(leff, transfer_nopure, 0, lmax, field_pairs,
                          file_name)
plt.close()
plt.clf()
print(f"  PLOT SAVED {plot_dir}/transfer_nopure.pdf")

print("   1G. Compute full couplings")
_, bpw_msk_nopure = get_inv_coupling(
    mask, nmt_bins, wcs=wcs, return_bp_win=True
)
_, bpw_msk_pure = get_inv_coupling(
    mask, nmt_bins, nmt_purify=True, wcs=wcs, return_bp_win=True
)
_, bpw_fil_nopure = get_inv_coupling(
    mask, nmt_bins, transfer=transfer_nopure["full_tf"], nmt_purify=False,
    wcs=wcs, return_bp_win=True
)
_, bpw_fil_pure = get_inv_coupling(
    mask, nmt_bins, transfer=transfer_pure["full_tf"], nmt_purify=True,
    wcs=wcs, return_bp_win=True
)
_, bpw_fil_pure_tf_nopure = get_inv_coupling(
    mask, nmt_bins, transfer=transfer_nopure["full_tf"], nmt_purify=True,
    wcs=wcs, return_bp_win=True
)
clth_msk_nopure = pu.bin_theory_cls(clth, bpw_msk_nopure)
clth_msk_pure = pu.bin_theory_cls(clth, bpw_msk_pure)
clth_fil_nopure = pu.bin_theory_cls(clth, bpw_fil_nopure)
clth_fil_pure = pu.bin_theory_cls(clth, bpw_fil_pure)
clth_fil_pure_tf_nopure = pu.bin_theory_cls(clth, bpw_fil_pure_tf_nopure)

fname = f"{plot_dir}/clth_comparison.pdf"
plt.plot(leff-3, clth_msk_nopure["BB"], "b.", alpha=0.4, label="msk nopure")
plt.plot(leff-1.5, clth_msk_pure["BB"], "c.", alpha=0.4, label="msk pure")
plt.plot(leff-0.5, clth_fil_nopure["BB"], "r.", alpha=0.4, label="fil nopure")
plt.plot(leff+0.5, clth_fil_pure["BB"], "y.", alpha=0.4, label="fil pure")
plt.plot(leff+1.5, clth_fil_pure_tf_nopure["BB"], "g.", alpha=0.4, label="fP (TF nopure)")  # noqa
plt.plot(leff+3, nmt_bins.bin_cell(clbb_in), "r--", alpha=0.2, label="Theory")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell^{BB}$")
plt.ylim((0, 3e-6))
plt.xlim((0, 100))
plt.legend(fontsize=14)
plt.savefig(fname)

print(f"  PLOT SAVED {fname}")
plt.close()


# Compute CL for unfiltered, masked, non-purified CMB EB power spectra
print("  2. Unfiltered, masked, non-purified CMB EB power spectra")

fname = f"{out_dir}/cls_masked_nopure.npz"
if os.path.isfile(fname):
    cls_masked_nopure = np.load(fname, allow_pickle=True)['cls']
else:
    cls_masked_nopure = []
    for i in range(nsims_cmb):
        if i % 50 == 0:
            print("   ", i)
        mp = load_cmb_sim(i, filtered=False)
        cl = compute_pspec(mp, mask, wcs=wcs)
        cls_masked_nopure.append(cl)
    cls_masked_nopure = np.array(cls_masked_nopure)
    np.savez(fname, cls=cls_masked_nopure)


# Compute CL for filtered, masked, non-purified CMB EB power spectra
print("  3. Filtered, masked, non-purified CMB EB power spectra")

fname = f"{out_dir}/cls_filtered_nopure.npz"
if os.path.isfile(fname):
    cls_filtered_nopure = np.load(fname, allow_pickle=True)["cls"]
else:
    cls_filtered_nopure = []
    for i in range(nsims_cmb):
        if i % 50 == 0:
            print("   ", i)
        mp = load_cmb_sim(i, filtered=True)
        cl = compute_pspec(
            mp, mask, transfer=transfer_nopure["full_tf"], wcs=wcs
        )
        cls_filtered_nopure.append(cl)
    cls_filtered_nopure = np.array(cls_filtered_nopure)
    np.savez(fname, cls=cls_filtered_nopure)


# Compute CL for unfiltered, masked, purified CMB EB power spectra
print("  4. Unfiltered, masked, purified CMB EB power spectra")

fname = f"{out_dir}/cls_masked_pure.npz"
if os.path.isfile(fname):
    cls_masked_pure = np.load(fname, allow_pickle=True)['cls']
else:
    cls_masked_pure = []
    for i in range(nsims_cmb):
        if i % 50 == 0:
            print("   ", i)
        mp = load_cmb_sim(i, filtered=False)
        cl = compute_pspec(mp, mask, nmt_purify=True, wcs=wcs)
        cls_masked_pure.append(cl)
    cls_masked_pure = np.array(cls_masked_pure)
    np.savez(fname, cls=cls_masked_pure)


# Compute pseudo_CL for filtered, masked, purified CMB EB power spectra
print("  5. Filtered, masked, purified CMB EB power spectra")

# print("  5A. TF without purification")
# fname = f"{out_dir}/cls_filtered_pure_tf_nopure.npz"
# if os.path.isfile(fname):
#     cls_filtered_pure_tf_nopure = np.load(fname, allow_pickle=True)['cls']
# else:
#     cls_filtered_pure_tf_nopure = []
#     for i in range(nsims_cmb):
#         if i % 50 == 0:
#             print("   ", i)
#         mp = load_cmb_sim(i, filtered=True)
#         cl = compute_pspec(mp, mask, transfer=transfer_nopure["full_tf"],
#                            wcs=wcs, nmt_purify=True)
#         cls_filtered_pure_tf_nopure.append(cl)
#     cls_filtered_pure_tf_nopure = np.array(cls_filtered_pure_tf_nopure)
#     np.savez(fname, cls=cls_filtered_pure_tf_nopure)

# print("  5B. TF with purification")
fname = f"{out_dir}/cls_filtered_pure.npz"
if os.path.isfile(fname):
    cls_filtered_pure = np.load(fname, allow_pickle=True)['cls']
else:
    cls_filtered_pure = []
    for i in range(nsims_cmb):
        if i % 50 == 0:
            print("   ", i)
        mp = load_cmb_sim(i, filtered=True)
        cl = compute_pspec(mp, mask, transfer=transfer_pure["full_tf"],
                           wcs=wcs, nmt_purify=True)
        cls_filtered_pure.append(cl)
    cls_filtered_pure = np.array(cls_filtered_pure)
    np.savez(fname, cls=cls_filtered_pure)


# Compute pseudo_CL for unfiltered, masked, non-purified CMB B-only power
# spectra
print("  6. Unfiltered, masked, non-purified CMB B-only power spectra")

fname = f"{out_dir}/cls_masked_noe.npz"
if os.path.isfile(fname):
    cls_masked_noe = np.load(fname, allow_pickle=True)['cls']
else:
    cls_masked_noe = []
    for i in range(nsims_cmb):
        if i % 50 == 0:
            print("   ", i)
        mp = load_cmb_sim(i, pols_keep="B")
        cl = compute_pspec(mp, mask, wcs=wcs)
        cls_masked_noe.append(cl)
    cls_masked_noe = np.array(cls_masked_noe)
    np.savez(fname, cls=cls_masked_noe)


# Compute pseudo_CL for filtered, masked, non-purified CMB B-only power spectra
print("  7. Filtered, masked, non-purified CMB B-only power spectra")

fname = f"{out_dir}/cls_filtered_noe.npz"
# if os.path.isfile(fname):
#     cls_filtered_noe = np.load(fname, allow_pickle=True)['cls']
# else:
cls_filtered_noe = []
for i in range(nsims_cmb):
    if i % 50 == 0:
        print("   ", i)
    mp = load_cmb_sim(i, filtered=True, pols_keep="B")
    cl = compute_pspec(
        mp, mask, wcs=wcs, transfer=transfer_nopure["full_tf"]
    )
    cls_filtered_noe.append(cl)
    print("cl", cl)
cls_filtered_noe = np.array(cls_filtered_noe)
np.savez(fname, cls=cls_filtered_noe)


# Map B-residuals (B-only component of filtered, mask-purified pure-E
# simulations)
print("  8. Mapping B-residuals from filtered pure-E sims")

os.makedirs(f"{out_dir}/sims_filt", exist_ok=True)
mp_sims = []

for i in range(nsims_purify):
    if i % 50 == 0:
        print("   ", i)
    fname = f"{out_dir}/sims_filt/sim_pureE_alpha2_filt_pure_Bout_{i:04d}.fits"
    if os.path.isfile(fname):
        mp = mu.read_map(
            fname,
            pix_type=pix_type,
            fields_hp=[0, 1, 2],
            car_template=car_template
        )
    else:
        # Load filtered pure-E sim
        mp = load_purification_sim(i, filtered=True)
        if i == 0:
            mu.plot_map(
                mp,
                file_name=f"{plot_dir}/sim_pureE_alpha2_filt_{i:04d}",
                pix_type="car"
            )
        # Mask-purify it
        mp = get_masked_map(mask, mp, nmt_purify=True)
        if i == 0:
            mu.plot_map(
                mp,
                file_name=f"{plot_dir}/sim_pureE_alpha2_filt_msk_purify_{i:04d}",  # noqa
                pix_type="car"
            )
        # Extract B-modes and remap them
        _, _, almB = mu.map2alm(mp, pix_type=pix_type)
        print(np.array([0*almB, 0*almB, almB]).dtype)
        mp = np.array(mu.alm2map(np.array([0*almB, 0*almB, almB]),
                                 pix_type=pix_type,
                                 car_map_template=car_template))
        mp = np.flip(mp, axis=(2,))  # weird CAR ordering
        if i == 0:
            mu.plot_map(
                mp,
                file_name=f"{plot_dir}/sim_pureE_alpha2_filt_pure_Bout_{i:04d}",  # noqa
                pix_type="car"
            )
        if pix_type == "car":
            mp = enmap.ndmap(mp, wcs=wcs)
        mu.write_map(fname, mp, pix_type=pix_type)

    mp_sims.append(mp)
mp_sims = np.array(mp_sims)


# Save M_ij = s_ipn *s_jpn, where s is the simulation vector of B-residuals
print("  9. Making deprojection matrix from mapped B-residual sims")

fname = f"{out_dir}/matcorr_alpha2_filt_pure_Bout.npz"
if os.path.isfile(fname):
    mat = np.load(fname, allow_pickle=True)['mat']
else:
    mat = []
    for i, s in enumerate(mp_sims):
        if i % 10 == 0:
            print("   ", i)
        if pix_type == "hp":
            mat.append(np.sum(mp_sims*s[None, :, :], axis=(1, 2)))
        else:  # car
            mat.append(np.sum(mp_sims*s[None, :, :, :], axis=(1, 2, 3)))
    mat = np.array(mat)
    np.savez(fname, mat=mat)


# Visualize eigenvalues of M
w, v = np.linalg.eigh(mat)
plt.plot(w[::-1])
plt.yscale('log')
plt.savefig(f"{plot_dir}/M_deproject_eigvals.pdf")
print(f"  PLOT SAVED {plot_dir}/M_deproject_eigvals.pdf")
plt.close()


def deproject_many(mp, simvec, mat):
    """
    Deproject B-residuals from a given map. This is achieved by computing

        mp_deproj = mp - simvec^T * pinv(N) * simvec * mp

    where pinv is the Moore-Penrose pseudoinverse, and N is the
    (k, k)-dimensional truncated version of mat (the deprojection matrix).

    Parameters
    ----------
        mp: np.array, shape [3, [map_shape]]
            Polarization map to deproject B-mode residuals from.
        simvec: np.array, shape [ndeproj, 3, [map_shape]]
            Simulated B-mode residuals to deproject.
        mat: np.array, shape [nsims, nsims]
            Projection matrix. Built from a number nsims >= ndeproj of
            B-mode residual sims.

    Results
    -------
        mp_deproj: np.array
            Deprojected map
    """
    # [3*npix]
    npix = len(mp[0].flatten())
    mp_h = mp.flatten()
    # [ndeproj, 3*npix]
    ndeproj = len(simvec)  # ndeproj is the number of sims to deproject
    simvec_h = simvec.reshape([ndeproj, 3*npix])

    # [ndeproj, ndeproj]
    Nij = mat[:ndeproj][:, :ndeproj]
    # [ndeproj]
    prods = np.sum(simvec_h*mp_h[None, :], axis=-1)
    # [ndeproj]
    Niprod = np.dot(pinvh(Nij, rtol=1E-3), prods)
    # Niprod = np.linalg.solve(Nij, prods)
    # [ndeproj, 3*npix] * [ndeproj] = [3*npix]
    mcont = np.dot(Niprod, simvec_h)
    # [3, npix]
    mp_deproj = (mp_h - mcont).reshape(mp.shape)

    return mp_deproj


# Deproject B-residuals from filtered, masked, nmt-purified CMB EB simulations
print("  10. Deprojecting B-residuals from CMB EB sims")

fname = f"{out_dir}/cls_filtered_pure_deproj.npz"
if os.path.isfile(fname):
    cls_filtered_pure_dep = np.load(fname, allow_pickle=True)['cls']
else:
    cls_filtered_pure_dep = []
    for i in range(nsims_deproj):
        if i % 10 == 0:
            print("   ", i)
        mp = load_cmb_sim(i, filtered=True)
        mp_masked = get_masked_map(mask, mp, nmt_purify=True)
        mp_masked_dep = deproject_many(mp_masked, mp_sims, mat)
        if i == 0:
            mu.plot_map(
                mp_masked_dep,
                file_name=f"{plot_dir}/sim_cmb_filt_purify_dep_{i:04d}",
                pix_type="car"
            )
        cl = compute_pspec(
            mp_masked_dep, mask, transfer=transfer_pure["full_tf"],
            wcs=wcs, nmt_purify=True
        )
        cls_filtered_pure_dep.append(cl)
    cls_filtered_pure_dep = np.array(cls_filtered_pure_dep)
    np.savez(fname, cls=cls_filtered_pure_dep)


# Plotting
print("  11. Plotting")
os.makedirs(f"{out_dir}/plots", exist_ok=True)


# cls_dict = {
#     "cls_filtered_noe": cls_filtered_noe,
#     "cls_filtered_nopure": cls_filtered_nopure,
#     "cls_filtered_pure": cls_filtered_pure,
#     "cls_filtered_pure_dep": cls_filtered_pure_dep,
#     "cls_masked_noe": cls_masked_noe,
#     "cls_masked_nopure": cls_masked_nopure,
#     "cls_masked_pure": cls_masked_pure,
# }


# for lab, cls in cls_dict.items():
#     print(lab, np.mean(np.array([cl["BB"] for cl in cls]), axis=(0,1)))
# for lab, cls in [cls_filtered_noe]
# print(np.mean(np.array))

# import sys; sys.exit()

# TODO: Load cls nested array in the correct way.
# print("cls_filtered_noe", np.array([cl["BB"] for cl in cls_filtered_noe]))

# plt.title('Cl, w. filtering, TFed')
# # plt.plot(leff, clbb_in, 'ro', label='Input')
# plt.plot(leff,
#          np.mean(np.array([cl["BB"] for cl in cls_filtered_noe]), axis=0),
#          'b:', label='Perfect purification')
# plt.plot(leff,
#          np.mean(np.array([cl["BB"] for cl in cls_filtered_nopure]), axis=0),
#          'k-', label='No purification')
# plt.plot(leff,
#          np.mean(np.array([cl["BB"] for cl in cls_filtered_pure]), axis=0),
#          'y--', label='Mask purification')
# plt.plot(leff,
#          np.mean(np.array([cl["BB"] for cl in cls_filtered_pure_dep]), axis=0),
#          'g.', label='Mask+sim purification')
# plt.xlim([30, lmax_plot])
# plt.xlabel('$\\ell$', fontsize=16)
# plt.ylabel('$C_\\ell$', fontsize=16)
# plt.yscale('log')
# #plt.ylim([1E-8, 3E-6])
# plt.yscale('log')
# plt.legend()
# plt.savefig(f"{plot_dir}/cl_filtered.pdf")
# plt.close()

# plt.figure()
# plt.title('sigma, w. filtering, TFed')
# plt.plot(leff,
#          np.std(np.array([cl["BB"] for cl in cls_filtered_noe]), axis=0),
#          'b:', label='Perfect purification')  # noqa
# plt.plot(leff,
#          np.std(np.array([cl["BB"] for cl in cls_filtered_nopure]), axis=0),
#                 'k-', label='No purification')  # noqa
# plt.plot(leff, np.std(cls_filtered_pure, axis=0)[3], 'y--', label='Mask purification')  # noqa
# plt.plot(leff, np.std(cls_filtered_pure_dep, axis=0)[3], 'g.', label='Mask+sim purification')  # noqa
# plt.xlim([30, lmax_plot])
# plt.xlabel('$\\ell$', fontsize=16)
# plt.ylabel('$\\sigma(C_\\ell)$', fontsize=16)
# plt.yscale('log')
# # plt.ylim([1E-9, 1E-5])
# plt.legend()
# plt.savefig(f"{plot_dir}/sigma_cl_filtered.pdf")
# plt.close()


plt.figure()
plt.title('Cl, no filtering')

y = np.mean(np.array([cl["BB"] for cl in cls_masked_noe]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_masked_noe]), axis=0)
plt.plot(leff, y, color='b', ls="-", label="Masked CMB, B only")
plt.fill_between(leff, y-yerr, y+yerr, color='b', alpha=0.2)

y = np.mean(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)
plt.plot(leff, y, color='k', ls="-", label="Masked CMB, no purification")
plt.fill_between(leff, y-yerr, y+yerr, color='k', alpha=0.2)

y = np.mean(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)
plt.plot(leff, y, color='y', ls="-", label="Masked CMB, purified")
plt.fill_between(leff, y-yerr, y+yerr, color='y', alpha=0.2)
plt.plot(clbb_in, 'r--', alpha=0.5, label="Theory")

plt.xlim([2, lmax_plot])
plt.ylim((1e-8, 1e-2))
plt.xlabel(r"$\ell$", fontsize=14)
plt.ylabel(r"$C_\ell^{BB}$", fontsize=14)
plt.yscale('log')
plt.legend()
print(f"  PLOT SAVED {plot_dir}/cl_masked.pdf")
plt.savefig(f"{plot_dir}/cl_masked.pdf")
plt.close()


thbb = nmt_bins.bin_cell(clbb_in)
y = np.mean(np.array([cl["BB"] for cl in cls_masked_noe]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_masked_noe]), axis=0)
plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
         'b-', alpha=0.5, label="Masked CMB, B only")

thbb = clth_msk_nopure["BB"]
y = np.mean(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)
plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
         'k-', alpha=0.5, label="Masked CMB, no purification")

thbb = clth_msk_pure["BB"]
y = np.mean(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)
plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
         'y-', alpha=0.5, label="Masked CMB, purified")
plt.axhline(0, color="r", ls="--")

plt.xlim([2, lmax_plot])
plt.ylim((-20, 20))
plt.xlabel(r"$\ell$", fontsize=14)
plt.ylabel(r"$(\hat{C}_\ell^{BB} - C_\ell^{BB,\, th})/(\sigma(C_\ell^{BB})/\sqrt{N_{\rm sims}})$",  # noqa
           fontsize=14)
plt.legend()
print(f"  PLOT SAVED {plot_dir}/cl_bias_masked.pdf")
plt.savefig(f"{plot_dir}/cl_bias_masked.pdf")
plt.close()


yerr_ref = np.std(np.array([cl["BB"] for cl in cls_masked_noe]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)
plt.plot(leff, yerr/yerr_ref, 'b-', alpha=0.5, label="Masked CMB, purified")

yerr = np.std(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)
plt.plot(leff, yerr/yerr_ref, 'k-', alpha=0.5,
         label="Masked CMB, no purification")

plt.axhline(1, color="k")

plt.xlim([2, lmax_plot])
plt.ylim((0.5, 1000))
plt.xlabel(r"$\ell$", fontsize=14)
plt.ylabel(r"$\sigma(C_\ell^{BB, X})/\sigma(C_\ell^{BB,\,\rm{B only}})$",
           fontsize=14)
plt.yscale('log')
plt.legend()
print(f"  PLOT SAVED {plot_dir}/cl_error_masked.pdf")
plt.savefig(f"{plot_dir}/cl_error_masked.pdf")
plt.close()


plt.title('Cl, w/ filtering')

y = np.mean(np.array([cl["BB"] for cl in cls_filtered_noe]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_filtered_noe]), axis=0)
plt.plot(leff, y, color='b', ls="-", label="Filtered CMB, B only")
plt.fill_between(leff, y-yerr, y+yerr, color='b', alpha=0.2)

y = np.mean(np.array([cl["BB"] for cl in cls_filtered_nopure]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_filtered_nopure]), axis=0)
plt.plot(leff, y, color='k', ls="-", label="Filtered CMB, no purification")
plt.fill_between(leff, y-yerr, y+yerr, color='k', alpha=0.2)

y = np.mean(np.array([cl["BB"] for cl in cls_filtered_pure]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_filtered_pure]), axis=0)
plt.plot(leff, y, color='y', ls="-", label="Filtered CMB, purified")
plt.fill_between(leff, y-yerr, y+yerr, color='y', alpha=0.2)

y = np.mean(np.array([cl["BB"] for cl in cls_filtered_pure_dep]), axis=0) * 10
yerr = np.std(np.array([cl["BB"] for cl in cls_filtered_pure_dep]), axis=0) * 10  # noqa
plt.plot(leff, y, color='g', ls="-", label="fCMB, purified & deproj.")
plt.fill_between(leff, y-yerr, y+yerr, color='g', alpha=0.2)

# y = np.mean(np.array([cl["BB"] for cl in cls_filtered_pure_tf_nopure]), axis=0)
# yerr = np.std(np.array([cl["BB"] for cl in cls_filtered_pure_tf_nopure]), axis=0)
# plt.plot(leff, y, color='darkorange', ls="-", label="fCMB, purified (TF nopure)")
# plt.fill_between(leff, y-yerr, y+yerr, color='darkorange', alpha=0.2)

plt.fill_between(leff, y-yerr, y+yerr, color='y', alpha=0.2)
plt.plot(clbb_in, 'r--', alpha=0.5, label="Theory")

plt.xlim([2, lmax_plot])
plt.ylim((1e-8, 1e-2))
plt.xlabel(r"$\ell$", fontsize=14)
plt.ylabel(r"$C_\ell^{BB}$", fontsize=14)
plt.yscale('log')
plt.legend()
print(f"  PLOT SAVED {plot_dir}/cl_filtered.pdf")
plt.savefig(f"{plot_dir}/cl_filtered.pdf")
plt.close()


thbb = nmt_bins.bin_cell(clbb_in)
y = np.mean(np.array([cl["BB"] for cl in cls_filtered_noe]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_filtered_noe]), axis=0)
plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
         'b-', alpha=0.5, label="Filtered CMB, B only")

thbb = clth_fil_nopure["BB"]
y = np.mean(np.array([cl["BB"] for cl in cls_filtered_nopure]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_filtered_nopure]), axis=0)
plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
         'k-', alpha=0.5, label="Filtered CMB, no purification")

thbb = clth_fil_pure["BB"]
y = np.mean(np.array([cl["BB"] for cl in cls_filtered_pure]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_filtered_pure]), axis=0)
plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
         'y-', alpha=0.5, label="Filtered CMB, purified")

y = np.mean(np.array([cl["BB"] for cl in cls_filtered_pure_dep]), axis=0) * 10
yerr = np.std(np.array([cl["BB"] for cl in cls_filtered_pure_dep]), axis=0) * 10  # noqa
plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
         'g-', alpha=0.5, label="fCMB, purified & deproj.")
plt.axhline(0, color="r", ls="--")

plt.xlim([2, lmax_plot])
plt.ylim((-20, 20))
plt.xlabel(r"$\ell$", fontsize=14)
plt.ylabel(r"$(\hat{C}_\ell^{BB} - C_\ell^{BB,\, th})/(\sigma(C_\ell^{BB})/\sqrt{N_{\rm sims}})$",  # noqa
           fontsize=14)
plt.legend()
print(f"  PLOT SAVED {plot_dir}/cl_bias_filtered.pdf")
plt.savefig(f"{plot_dir}/cl_bias_filtered.pdf")
plt.close()


yerr_ref = np.std(np.array([cl["BB"] for cl in cls_filtered_noe]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_filtered_nopure]), axis=0)
plt.plot(leff, yerr/yerr_ref, 'k-', alpha=0.5,
         label="Filtered CMB, no purification")
yerr = np.std(np.array([cl["BB"] for cl in cls_filtered_pure]), axis=0)
plt.plot(leff, yerr/yerr_ref, 'y-', alpha=0.5, label="Filtered CMB, purified")
yerr = np.std(np.array([cl["BB"] for cl in cls_filtered_pure_dep]), axis=0)
plt.plot(leff, yerr/yerr_ref, 'g-', alpha=0.5,
         label="fCMB, purified & deproj.")

plt.axhline(1, color="k")

plt.xlim([2, lmax_plot])
plt.ylim((0.5, 1000))
plt.xlabel(r"$\ell$", fontsize=14)
plt.ylabel(r"$\sigma(C_\ell^{BB, X})/\sigma(C_\ell^{BB,\,\rm{B only}})$",
           fontsize=14)
plt.yscale('log')
plt.legend()
print(f"  PLOT SAVED {plot_dir}/cl_error_filtered.pdf")
plt.savefig(f"{plot_dir}/cl_error_filtered.pdf")
plt.close()
