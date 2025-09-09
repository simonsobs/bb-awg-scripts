import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
import os
import healpy as hp
from pixell import enmap, utils
import camb

from soopercool import ps_utils as pu
from soopercool import map_utils as mu

"""
Simplified version of "iterative_purification", but ignoring filtering.
"""


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


def get_inv_coupling(mask, nmt_bins, transfer=None, nmt_purify=False,
                     return_bp_win=False, wcs=None, lmax_mask=None,
                     overwrite_coupling=False):
    """
    """
    tf_correct = transfer is not None
    pix_type = "hp" if wcs is None else "car"

    pure_label = "_nmt_purify" if nmt_purify else ""
    tf_label = "_tf_correct" if tf_correct else ""
    coupling_fname = f"{out_dir}/inv_coupling{tf_label}{pure_label}_{pix_type}.npz"

    if os.path.isfile(coupling_fname) and not overwrite_coupling:
        inv_coupling = np.load(coupling_fname)["inv_coupling"]
        if not return_bp_win:
            return inv_coupling
        else:
            try:
                bp_win = np.load(coupling_fname)["bp_win"]
                return inv_coupling, bp_win
            except KeyError:
                pass

    lmax = nmt_bins.lmax if lmax_mask is None else lmax_mask
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
    # print(f"     COUPLINGS SAVED {coupling_fname}")

    if return_bp_win:
        return inv_coupling, bp_win
    return inv_coupling


print("  Reading inputs")

# pixelization related
nside = None
pix_type = "car"
pix_lab = "car" if pix_type == "car" else f"nside{nside}"
res_arcmin = 20
car_template = "/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/band_car_fejer1_20arcmin.fits"  # "/home/kw6905/bbdev/bb-awg-scripts/pipeline/simpure/band_car_fejer1_20arcmin.fits"  # noqa
beam_fwhm = 30

if pix_type == "car":
    if car_template is not None:
        geometry = enmap.read_map_geometry(car_template)
        wcs = geometry[1]
        res = np.min(np.abs(wcs.wcs.cdelt)) * 60.
    elif res_arcmin is not None:
        res = res_arcmin * np.pi/180/60
        geometry = enmap.fullsky_geometry(res=res_arcmin*utils.arcmin)
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
if pix_type == "car":
    binning_file = "/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/binning_car_lmax540_deltal10.npz"  # noqa
    binning = np.load(binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"], binning["bin_high"] + 1)
else:
    nmt_bins = nmt.NmtBin.from_nside_linear(nside, 10)
leff = nmt_bins.get_effective_ells()
print("   leff", leff)

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
nsims_cmb = 20  # number of validation sims

out_dir = "."  ## YOUR OUTPUT DIR
plot_dir = f"{out_dir}/plots"
os.makedirs(plot_dir, exist_ok=True)
lmax_plot = 300
overwrite = False  # If True, always recompute products.


# apodized mask
mask_file = "."  ## YOUR INPUT MASK
mask = mu.read_map(mask_file,
                   pix_type=pix_type,
                   car_template=car_template)
mask_bin = (mask > 0).astype(float)
if pix_type == "car":
    mask_bin = enmap.ndmap(mask_bin, wcs=wcs)
mu.plot_map(mask, file_name=f"{plot_dir}/mask",
            pix_type=pix_type)
print(f"     PLOT {plot_dir}/mask.png")
mu.plot_map(mask_bin, file_name=f"{plot_dir}/mask_bin",
            pix_type=pix_type)
print(f"     PLOT {plot_dir}/mask_bin.png")


def get_masked_map(mask, mp, nmt_purify=False, pix_type=None):
    """
    """
    assert mp.shape[0] in [2, 3]
    wcs = None if pix_type == "hp" else mask.wcs
    shape = mask.shape

    if not nmt_purify:
        if pix_type == "hp":
            return mp*mask
        else:  # car
            return enmap.ndmap(mp*mask[None, :, :], wcs=wcs)
    else:
        f = nmt.NmtField(mask, mp[-2:], purify_b=True, wcs=wcs)
        mp_masked = np.array([m.reshape(shape) for m in f.get_maps()])
        if mp.shape[0] == 3:  # TQU maps
            mp_masked = np.array([mp[0]*mask, mp_masked[0], mp_masked[1]])
        if pix_type == "hp":
            return mp_masked
        else:  # car
            # FIXING BUG: f.get_maps() doesn't preserve the wcs convention
            mp_masked = np.flip(mp_masked, axis=(1,2))
            return enmap.ndmap(mp_masked, wcs=wcs)
        

def load_cmb_sim(sim_id, filtered=False, pols_keep="EB"):
    """
    """
    base_dir = "/cephfs/soukdata/user_data/kwolz/simpure"
    suffix = "_f090_science_filtered" if filtered else ""
    sim_dir = "filtered_cmb_sims/satp3/f090/butter4_cutoff_1e-2/coadded_sims"
    if not filtered:
        sim_dir = "cmb_sims"
    res_str = "20arcmin" if pix_type == "car" else f"nside{nside}"  # add by hand. TODO: generalize  # noqa
    sim_fname = f"cmb{pols_keep}_{res_str}_fwhm30.0_sim{sim_id:04d}_{pix_type.upper()}{suffix}.fits"  # noqa

    map =  mu.read_map(
        f"{base_dir}/{sim_dir}/{sim_fname}",
        pix_type=pix_type,
        fields_hp=[0, 1, 2],
        car_template=car_template,
        convert_K_to_muK=True
    )
    return get_masked_map(mask_bin, map, pix_type=pix_type)


def compute_pspec(map, mask, nmt_bins, transfer=None, nmt_purify=False,
                  map2=None, wcs=None, return_just_fields=False,
                  nmt_purify_mcm=None, masked_on_input=False):
    """
    """
    lmax = nmt_bins.lmax
    field = {
        "spin0": nmt.NmtField(mask, map[:1], wcs=wcs, lmax=lmax,
                              masked_on_input=masked_on_input),
        "spin2": nmt.NmtField(mask, map[1:], wcs=wcs, purify_b=nmt_purify,
                              lmax=lmax, masked_on_input=masked_on_input)
    }
    map2 = map
    field2 = field

    if map2 is not None:
        field2 = {
            "spin0": nmt.NmtField(mask, map2[:1], wcs=wcs, lmax=lmax,
                                  masked_on_input=masked_on_input),
            "spin2": nmt.NmtField(mask, map2[1:], wcs=wcs, purify_b=nmt_purify,
                                  lmax=lmax, masked_on_input=masked_on_input)
        }
    if return_just_fields:
        return field, field2

    pcls = pu.get_coupled_pseudo_cls(field, field2, nmt_bins)
    if nmt_purify_mcm is None:
        nmt_purify_mcm = nmt_purify
    inv_coupling = get_inv_coupling(
        mask, nmt_bins, transfer=transfer, nmt_purify=nmt_purify_mcm, wcs=wcs
    )

    return pu.decouple_pseudo_cls(pcls, inv_coupling)


print("  Compute full couplings")
_, bpw_msk_nopure = get_inv_coupling(
    mask, nmt_bins, wcs=wcs, return_bp_win=True,
)
_, bpw_msk_pure = get_inv_coupling(
    mask, nmt_bins, nmt_purify=True, wcs=wcs, return_bp_win=True
)
clth_msk_nopure = pu.bin_theory_cls(clth, bpw_msk_nopure)
clth_msk_pure = pu.bin_theory_cls(clth, bpw_msk_pure)

plt.clf()
plt.figure()
fname = f"{plot_dir}/clth_comparison.pdf"
plt.plot(leff-3, clth_msk_nopure["BB"], "k.", alpha=0.4, label="msk nopure")
plt.plot(leff-1.5, clth_msk_pure["BB"], "b.", alpha=0.4, label="msk pure")
plt.plot(leff+3, nmt_bins.bin_cell(clbb_in), "r--", alpha=0.2, label="Theory")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell^{BB}$")
plt.ylim((0, 3e-6))
plt.xlim((0, 100))
plt.legend(fontsize=14)
plt.savefig(fname)

print(f"    PLOT SAVED {fname}")
plt.close()


# Compute CLs
print("  Power spectrum computation")
print("  - Masked cmbEB without purification")
fname = f"{out_dir}/cls_masked_nopure.npz"
if os.path.isfile(fname) and not overwrite:
    cls_masked_nopure = np.load(fname, allow_pickle=True)['cls']
else:
    cls_masked_nopure = []
    for i in range(nsims_cmb):
        if i % 10 == 0:
            print("   ", i)
        mp = load_cmb_sim(i, filtered=False)  # noqa
        cl = compute_pspec(mp, mask, nmt_bins, wcs=wcs)
        cls_masked_nopure.append(cl)
    cls_masked_nopure = np.array(cls_masked_nopure)
    np.savez(fname, cls=cls_masked_nopure)

print("  - Masked cmbEB with purification")
fname = f"{out_dir}/cls_masked_pure.npz"
if os.path.isfile(fname) and not overwrite:
    cls_masked_pure = np.load(fname, allow_pickle=True)['cls']
else:
    cls_masked_pure = []
    for i in range(nsims_cmb):
        if i % 10 == 0:
            print("   ", i)
        mp = load_cmb_sim(i, filtered=False)
        if i == 0:
            mu.plot_map(
                mp,
                file_name=f"{plot_dir}/sim_cmb_{i:04d}",
                pix_type=pix_type
            )
            mu.plot_map(
                get_masked_map(mask, mp, nmt_purify=False, pix_type=pix_type),
                file_name=f"{plot_dir}/sim_cmb_masked_{i:04d}",
                pix_type=pix_type
            )
            mu.plot_map(
                get_masked_map(mask, mp, nmt_purify=True, pix_type=pix_type),
                file_name=f"{plot_dir}/sim_cmb_masked_purif_{i:04d}",
                pix_type=pix_type
            )
        cl = compute_pspec(mp, mask, nmt_bins, nmt_purify=True, wcs=wcs)
        cls_masked_pure.append(cl)
    cls_masked_pure = np.array(cls_masked_pure)
    np.savez(fname, cls=cls_masked_pure)

print("  - Masked cmbB (without purification)")
fname = f"{out_dir}/cls_noe_masked.npz"
if os.path.isfile(fname) and not overwrite:
    cls_noe_masked = np.load(fname, allow_pickle=True)['cls']
else:
    cls_noe_masked = []
    for i in range(nsims_cmb):
        if i % 10 == 0:
            print("   ", i)
        mp = load_cmb_sim(i, pols_keep="B")
        cl = compute_pspec(mp, mask, nmt_bins, wcs=wcs)
        cls_noe_masked.append(cl)
    cls_noe_masked = np.array(cls_noe_masked)
    np.savez(fname, cls=cls_noe_masked)


# Plotting
print("  Plotting")
os.makedirs(f"{out_dir}/plots", exist_ok=True)

plt.figure()
plt.title('Cl, no filtering')

y = np.mean(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)
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
y = np.mean(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)
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
print(f"    PLOT SAVED {plot_dir}/cl_bias_masked.pdf")
plt.savefig(f"{plot_dir}/cl_bias_masked.pdf")
plt.close()


yerr_ref = np.std(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)
yerr = np.std(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)
plt.plot(leff, yerr/yerr_ref, 'b-', alpha=0.5, label="Masked CMB, purified")
sigma_masked = yerr/yerr_ref

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
print(f"    PLOT SAVED {plot_dir}/cl_error_masked.pdf")
plt.savefig(f"{plot_dir}/cl_error_masked.pdf")
plt.close()
