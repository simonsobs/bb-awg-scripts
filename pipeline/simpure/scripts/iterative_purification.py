import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinvh
import pymaster as nmt
import os
import healpy as hp
from pixell import enmap, utils
import camb

from soopercool import ps_utils as pu
from soopercool import map_utils as mu
from soopercool import coupling_utils as cu


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


def plot_transfer_function(lb, tf_dict, lmin, lmax, field_pairs, file_name):
    """
    Plot the transfer function given an input dictionary.
    """
    import copy
    npan = len(field_pairs)
    plt.figure(figsize=(25*npan/9, 25*npan/9))
    grid = plt.GridSpec(npan, npan, hspace=0.3, wspace=0.3)
    colors = ["navy", "darkorange", "red", "darkgreen"]

    cases = [""]
    if f"EE_to_EE" not in tf_dict:
        cases = list(tf_dict.keys())
    print("TF cases", cases)

    for id1, f1 in enumerate(field_pairs):
        for id2, f2 in enumerate(field_pairs):
            ax = plt.subplot(grid[id1, id2])
            expected = 1. if f1 == f2 else 0.
            ylims = [0, 1.05] if f1 == f2 else [-0.01, 0.01]

            ax.axhline(expected, color="k", ls="--", zorder=6)
            # We need to understand the offdigonal TF panels in the presence of
            # NaMaster purification - we don't have a clear interpretation.
            ax.set_title(f"{f1} $\\rightarrow$ {f2}", fontsize=14)
            
            for ic, case in enumerate(cases):
                if case == "":
                    tf = copy.deepcopy(tf_dict)
                else:
                    tf = copy.deepcopy(tf_dict[case])
                ax.plot(lb, tf[f"{f1}_to_{f2}"], color=colors[ic])

                if id1 == npan-1:
                    ax.set_xlabel(r"$\ell$", fontsize=14)
                else:
                    ax.set_xticks([])

                if f1 != f2:
                    ax.ticklabel_format(axis="y", style="scientific",
                                        scilimits=(0, 0), useMathText=True)
            ax.set_xlim(lmin, lmax)
            ax.set_ylim(ylims[0], ylims[1])
    for ic, c in enumerate(cases):
        if c != "":
            plt.plot([], [], label=c, color=colors[ic])
    if len(cases) > 0:
        plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
    plt.clf()


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


print("  0. Reading inputs")

# pixelization related
nside = 256
pix_type = "car"  # "hp"
pix_lab = "car" if pix_type == "car" else f"nside{nside}"
res_arcmin = 20
car_template = "/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/data/band_car_fejer1_20arcmin.fits"  # "/home/kw6905/bbdev/bb-awg-scripts/pipeline/simpure/band_car_fejer1_20arcmin.fits"  # noqa
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

if 180.*60/lmax > beam_fwhm:
    print("WARNING: beam resolution is lower than pixel scale. Consider "
          "increasing the beam FWHM.")
lmax_sim = lmax + 500


# binning
bin_label = ""
if pix_type == "car":
    binning_file = "/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/data/binning_car_lmax540_deltal10.npz"  # noqa
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

# general
nsims_purify = 300  # number of pure-E sims used for template deprojection
nsims_cmb = 100  # number of validation sims
nsims_transfer = 50  # number of (pureE, pureB) sims used for transfer function
id_sim_transfer_start = 0

apo_scale = 10
apo_type = "C1"
lmax = lmax//4
filter_setup = "butter4_20251007"
out_dir = f"/cephfs/soukdata/user_data/kwolz/simpure/purification/{filter_setup}_thresh20percent"  # noqa
plot_dir = f"{out_dir}/plots"
os.makedirs(plot_dir, exist_ok=True)
lmax_plot = 300
overwrite = False  # If True, always recompute products.
deproject_null = True  # Deproject null vector instead of pureB template.
ignore_filtering = False  # If True, only check mask-based purification.

# apodized mask
mask_file = f"/cephfs/soukdata/user_data/kwolz/simpure/filtered_pure_sims/satp3/f090/{filter_setup}/mask_thresh20percent/masks/analysis_mask_apo10_C1_car.fits"  # noqa
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
    sim_dir = f"filtered_cmb_sims/satp3/f090/{filter_setup}/coadded_sims"
    if not filtered:
        sim_dir = "cmb_sims"
    res_str = "20arcmin" if pix_type == "car" else f"nside{nside}"  # add by hand. TODO: generalize  # noqa
    sim_fname = f"cmb{pols_keep}_{res_str}_fwhm{beam_fwhm}.0_sim{sim_id:04d}_{pix_type.upper()}{suffix}.fits"  # noqa

    map =  mu.read_map(
        f"{base_dir}/{sim_dir}/{sim_fname}",
        pix_type=pix_type,
        fields_hp=[0, 1, 2],
        car_template=car_template,
        convert_K_to_muK=True
    )
    return get_masked_map(mask_bin, map, pix_type=pix_type)


def load_transfer_sim(sim_id, filtered=False, type=None):
    """
    """
    assert type in [f"pure{p}" for p in "TEB"], "Invalid pure type"
    base_dir = "/cephfs/soukdata/user_data/kwolz/simpure"
    suffix = "_f090_science_filtered" if filtered else ""
    sim_dir = f"filtered_pure_sims/satp3/f090/{filter_setup}/coadded_sims"  # noqa
    if not filtered:
        sim_dir = "input_sims"
    res_str = "20.0arcmin" if pix_type == "car" else f"nside{nside}"  # add by hand. TODO: generalize  # noqa
    sim_fname = f"{type}_{res_str}_fwhm{beam_fwhm}.0_sim{sim_id:04d}_{pix_type.upper()}{suffix}.fits"  # noqa

    map = mu.read_map(
        f"{base_dir}/{sim_dir}/{sim_fname}",
        pix_type=pix_type,
        fields_hp=[0, 1, 2],
        car_template=car_template,
        convert_K_to_muK=True
    )
    return get_masked_map(mask_bin, map, pix_type=pix_type)



def load_purification_sim(sim_id, filtered=False):
    """
    """
    return load_transfer_sim(sim_id, filtered=filtered, type="pureE")  # noqa


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


def extract_pure_mode(map, which_mode="B", pix_type="hp"):
    """
    """
    alms = {}
    alms["T"], alms["E"], alms["B"] = mu.map2alm(map, pix_type=pix_type)
    for p in "TEB":
        if p != which_mode:
            alms[p] *= 0.
    geometry = None if pix_type == "hp" else map.geometry

    return mu.alm2map(
        [alms["T"], alms["E"], alms["B"]], pix_type=pix_type,
        nside=nside, car_map_template=car_template
    )



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
        mp_deprojected: np.array
            Output map
        mp_deproject: np.array
            Map template to deproject from map
    """
    wcs = None if not hasattr(mp, "wcs") else mp.wcs
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
    mp_deproject = mcont.reshape(mp.shape)
    mp_deprojected = (mp_h - mcont).reshape(mp.shape)
    if wcs is not None:
        mp_deproject = enmap.ndmap(mp_deproject, wcs=wcs)
        mp_deprojected = enmap.ndmap(mp_deprojected, wcs=wcs)

    return mp_deprojected, mp_deproject


if not ignore_filtering:
    print("  1. Make deprojection matrix")
    if not deproject_null:
        # Map B-residuals (B-only component of filtered, mask-purified pure-E
        # simulations)
        print("  1A. Mapping B-residuals from filtered pure-E sims")

        os.makedirs(f"{out_dir}/sims_filt", exist_ok=True)
        mp_sims = []

        for i in range(nsims_purify):
            if i % 50 == 0:
                print("   ", i)
            fname = f"{out_dir}/sims_filt/sim_pureE_alpha2_filt_pure_Bout_{i:04d}.fits"
            if os.path.isfile(fname) and not overwrite:
                mp_masked_bonly = mu.read_map(
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
                        pix_type=pix_type
                    )
                # Mask-purify it
                mp_masked = get_masked_map(mask, mp, nmt_purify=True, pix_type=pix_type)
                if i == 0:
                    mu.plot_map(
                        mp_masked,
                        file_name=f"{plot_dir}/sim_pureE_alpha2_filt_msk_purify_{i:04d}",  # noqa
                        pix_type=pix_type
                    )
                # Extract B-modes and remap them
                mp_masked_bonly = extract_pure_mode(mp_masked, which_mode="B", pix_type=pix_type)
                if i == 0:
                    mu.plot_map(
                        mp_masked_bonly,
                        file_name=f"{plot_dir}/sim_pureE_alpha2_filt_pure_Bout_{i:04d}",  # noqa
                        pix_type=pix_type
                    )
                mu.write_map(fname, mp_masked_bonly, pix_type=pix_type)

            mp_sims.append(mp_masked_bonly)
        mp_sims = np.array(mp_sims)


        # Save M_ij = s_ipn *s_jpn, where s is the simulation vector of B-residuals
        print("  1B. Making deprojection matrix from mapped B-residual sims")

        fname = f"{out_dir}/matcorr_alpha2_filt_pure_Bout_nsims{nsims_purify}.npz"
        if os.path.isfile(fname) and not overwrite:
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
        plot_file = f"{plot_dir}/M_deproject_eigvals_nsims{nsims_purify}.pdf"
        plt.savefig(plot_file)
        print(f"    PLOT SAVED {plot_file}")
        plt.close()
    else:
        print("  1A. Null B-residuals to mock deproject")
        mp = load_cmb_sim(0, filtered=True)
        mp_sims = np.zeros((nsims_purify,) + mp.shape)
        mat = np.eye(nsims_purify)


    print("  2. Transfer function computation")

    print("  2A. TF sims unfiltered w/o purification")
    fname = out_dir + f"/cls_tf_unfiltered_nopure_nsims{nsims_transfer}.npz"
    if os.path.isfile(fname) and not overwrite:
        cls_tf_unfiltered_nopure = np.load(fname, allow_pickle=True)["cls"]
    else:
        cls_tf_unfiltered_nopure = []
        for i in range(id_sim_transfer_start,
                       id_sim_transfer_start+nsims_transfer):
            if i % 10 == 0:
                print("   ", i)
            fields, fields2 = ({}, {})
            for typ in [f"pure{f}" for f in "TEB"]:
                mp = load_transfer_sim(i, filtered=False, type=typ)  # noqa
                f1, f2 = compute_pspec(
                    mp, mask, nmt_bins, nmt_purify=False, wcs=wcs,
                    return_just_fields=True
                )
                fields[typ] = f1
                fields2[typ] = f2
            pcls_mat = pu.get_pcls_mat_transfer(fields, nmt_bins, fields2)
            cls_tf_unfiltered_nopure.append(pcls_mat)
        cls_tf_unfiltered_nopure = np.array(cls_tf_unfiltered_nopure)
        np.savez(fname, cls=cls_tf_unfiltered_nopure)

    print("  2B. TF sims unfiltered w/ purification")
    fname = out_dir + "/cls_tf_unfiltered_pure_nsims{nsims_transfer}.npz"
    if os.path.isfile(fname) and not overwrite:
        cls_tf_unfiltered_pure = np.load(fname, allow_pickle=True)["cls"]
    else:
        cls_tf_unfiltered_pure = []
        for i in range(id_sim_transfer_start,
                       id_sim_transfer_start+nsims_transfer):
            if i % 10 == 0:
                print("   ", i)
            fields, fields2 = ({}, {})
            for typ in [f"pure{f}" for f in "TEB"]:
                mp = load_transfer_sim(i, filtered=False, type=typ)  # noqa
                f1, f2 = compute_pspec(
                    mp, mask, nmt_bins, nmt_purify=True, wcs=wcs,
                    return_just_fields=True
                )
                fields[typ] = f1
                fields2[typ] = f2
            pcls_mat = pu.get_pcls_mat_transfer(fields, nmt_bins, fields2)
            cls_tf_unfiltered_pure.append(pcls_mat)
        cls_tf_unfiltered_pure = np.array(cls_tf_unfiltered_pure)
        np.savez(fname, cls=cls_tf_unfiltered_pure)

    print("  2C. TF sims filtered w/o purification")
    fname = out_dir + "/cls_tf_filtered_nopure_nsims{nsims_transfer}.npz"
    if os.path.isfile(fname) and not overwrite:
        cls_tf_filtered_nopure = np.load(fname, allow_pickle=True)["cls"]
    else:
        cls_tf_filtered_nopure = []
        for i in range(id_sim_transfer_start,
                       id_sim_transfer_start+nsims_transfer):
            if i % 10 == 0:
                print("   ", i)
            fields, fields2 = ({}, {})
            for typ in [f"pure{f}" for f in "TEB"]:
                mp = load_transfer_sim(i, filtered=True, type=typ)  # noqa
                f1, f2 = compute_pspec(
                    mp, mask, nmt_bins, nmt_purify=False, wcs=wcs,
                    return_just_fields=True
                )
                fields[typ] = f1
                fields2[typ] = f2
            pcls_mat = pu.get_pcls_mat_transfer(fields, nmt_bins, fields2)
            cls_tf_filtered_nopure.append(pcls_mat)
        cls_tf_filtered_nopure = np.array(cls_tf_filtered_nopure)
        np.savez(fname, cls=cls_tf_filtered_nopure)

    print("  2D. TF sims filtered w/ purification")
    fname = out_dir + "/cls_tf_filtered_pure_nsims{nsims_transfer}.npz"
    if os.path.isfile(fname) and not overwrite:
        cls_tf_filtered_pure = np.load(fname, allow_pickle=True)["cls"]
    else:
        cls_tf_filtered_pure = []
        for i in range(id_sim_transfer_start,
                       id_sim_transfer_start+nsims_transfer):
            if i % 10 == 0:
                print("   ", i)
            fields, fields2 = ({}, {})
            for typ in [f"pure{f}" for f in "TEB"]:
                mp = load_transfer_sim(i, filtered=True, type=typ)  # noqa
                f1, f2 = compute_pspec(
                    mp, mask, nmt_bins, nmt_purify=True,
                    wcs=wcs, return_just_fields=True
                )
                fields[typ] = f1
                fields2[typ] = f2
            pcls_mat = pu.get_pcls_mat_transfer(fields, nmt_bins, fields2)
            cls_tf_filtered_pure.append(pcls_mat)
        cls_tf_filtered_pure = np.array(cls_tf_filtered_pure)
        np.savez(fname, cls=cls_tf_filtered_pure)
    
    print("  2E. TF sims filtered w/ purification & deprojection")
    fname = out_dir + f"/cls_tf_filtered_pure_dep_nsims{nsims_purify}.npz"
    if os.path.isfile(fname) and not overwrite:
        cls_tf_filtered_pure_dep = np.load(fname, allow_pickle=True)["cls"]
    else:
        cls_tf_filtered_pure_dep = []
        for i in range(id_sim_transfer_start,
                       id_sim_transfer_start+nsims_transfer):
            if i % 10 == 0:
                print("   ", i)
            fields, fields2 = ({}, {})
            for typ in [f"pure{f}" for f in "TEB"]:
                mp = load_transfer_sim(i, filtered=True, type=typ)  # noqa
                mp_masked = get_masked_map(mask, mp, nmt_purify=True,
                                           pix_type=pix_type)
                mp_masked_dep, _ = deproject_many(mp_masked, mp_sims, mat)
                f1, f2 = compute_pspec(
                    mp_masked_dep, mask, nmt_bins, wcs=wcs,
                    return_just_fields=True,
                    masked_on_input=True
                )
                fields[typ] = f1
                fields2[typ] = f2
            pcls_mat = pu.get_pcls_mat_transfer(fields, nmt_bins, fields2)
            cls_tf_filtered_pure_dep.append(pcls_mat)
        cls_tf_filtered_pure_dep = np.array(cls_tf_filtered_pure_dep)
        np.savez(fname, cls=cls_tf_filtered_pure_dep)

    print("  2F. TF w/ purification")
    fp_dum = ("None", "None")
    pcls_mat_filt_dict = {fp_dum: {"filtered": cls_tf_filtered_pure}}
    mean_pcls_mat_filt_dict = {fp_dum: np.mean(cls_tf_filtered_pure, axis=0)}
    mean_pcls_mat_unfilt_dict = {fp_dum: np.mean(cls_tf_unfiltered_pure, axis=0)}
    transfer_pure = cu.get_transfer_dict(
        mean_pcls_mat_filt_dict, mean_pcls_mat_unfilt_dict, pcls_mat_filt_dict,
        [fp_dum]
    )[fp_dum]

    print("  2G. TF w/o purification")
    pcls_mat_filt_dict = {fp_dum: {"filtered": cls_tf_filtered_nopure}}
    mean_pcls_mat_filt_dict = {fp_dum: np.mean(cls_tf_filtered_nopure, axis=0)}
    mean_pcls_mat_unfilt_dict = {fp_dum: np.mean(cls_tf_unfiltered_nopure, axis=0)}
    transfer_nopure = cu.get_transfer_dict(
        mean_pcls_mat_filt_dict, mean_pcls_mat_unfilt_dict, pcls_mat_filt_dict,
        [fp_dum]
    )[fp_dum]

    print("  2H. TF with purification & deprojection")
    pcls_mat_filt_dict = {fp_dum: {"filtered": cls_tf_filtered_pure_dep}}
    mean_pcls_mat_filt_dict = {fp_dum: np.mean(cls_tf_filtered_pure_dep, axis=0)}
    mean_pcls_mat_unfilt_dict = {fp_dum: np.mean(cls_tf_unfiltered_pure, axis=0)}
    transfer_pure_dep = cu.get_transfer_dict(
        mean_pcls_mat_filt_dict, mean_pcls_mat_unfilt_dict, pcls_mat_filt_dict,
        [fp_dum]
    )[fp_dum]

    field_pairs = ["EE", "EB", "BE", "BB"]
    file_name = f"{plot_dir}/transfer_dep_nsims{nsims_purify}_tf_nsims{nsims_transfer}.pdf"  # noqa
    tf_dict = {
        "nopure": transfer_nopure, "pure": transfer_pure,
        f"pure_dep (N={nsims_purify})": transfer_pure_dep
    }

    plot_transfer_function(leff, tf_dict, 0, lmax_plot, field_pairs,
                           file_name=file_name)
    plt.close()
    plt.clf()
    print(f"    PLOT SAVED {file_name}")


print("  3. Compute full couplings")
_, bpw_msk_nopure = get_inv_coupling(
    mask, nmt_bins, wcs=wcs, return_bp_win=True,
)
_, bpw_msk_pure = get_inv_coupling(
    mask, nmt_bins, nmt_purify=True, wcs=wcs, return_bp_win=True
)
if not ignore_filtering:
    _, bpw_fil_nopure = get_inv_coupling(
        mask, nmt_bins, transfer=transfer_nopure["full_tf"], nmt_purify=False,
        wcs=wcs, return_bp_win=True, overwrite_coupling=True
    )
    _, bpw_fil_pure = get_inv_coupling(
        mask, nmt_bins, transfer=transfer_pure["full_tf"], nmt_purify=True,
        wcs=wcs, return_bp_win=True, overwrite_coupling=True
    )
    _, bpw_fil_pure_tf_nopure = get_inv_coupling(
        mask, nmt_bins, transfer=transfer_nopure["full_tf"], nmt_purify=True,
        wcs=wcs, return_bp_win=True, overwrite_coupling=True
    )
    _, bpw_fil_pure_dep = get_inv_coupling(
        mask, nmt_bins, transfer=transfer_pure_dep["full_tf"], nmt_purify=True,
        wcs=wcs, return_bp_win=True, overwrite_coupling=True
    )

clth_msk_nopure = pu.bin_theory_cls(clth, bpw_msk_nopure)
clth_msk_pure = pu.bin_theory_cls(clth, bpw_msk_pure)
if not ignore_filtering:
    clth_fil_nopure = pu.bin_theory_cls(clth, bpw_fil_nopure)
    clth_fil_pure = pu.bin_theory_cls(clth, bpw_fil_pure)
    clth_fil_pure_tf_nopure = pu.bin_theory_cls(clth, bpw_fil_pure_tf_nopure)
    clth_fil_pure_dep = pu.bin_theory_cls(clth, bpw_fil_pure_dep)

plt.clf()
plt.figure()
fname = f"{plot_dir}/clth_comparison.pdf"
plt.plot(leff-3, clth_msk_nopure["BB"], "k.", alpha=0.4, label="msk nopure")
plt.plot(leff-1.5, clth_msk_pure["BB"], "b.", alpha=0.4, label="msk pure")
if not ignore_filtering:
    plt.plot(leff-0.5, clth_fil_nopure["BB"], "r.", alpha=0.4, label="fil nopure")
    plt.plot(leff+0.5, clth_fil_pure["BB"], "y.", alpha=0.4, label="fil pure")
    plt.plot(leff+1.5, clth_fil_pure_dep["BB"], "g.", alpha=0.4,
             label=f"fil pure dep ({nsims_purify} sims)")

plt.plot(leff+3, nmt_bins.bin_cell(clth["BB"][:nmt_bins.lmax+1]), "r--", alpha=0.2, label="Theory")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell^{BB}$")
plt.ylim((0, 3e-6))
plt.xlim((0, 100))
plt.legend(fontsize=14)
plt.savefig(fname)

print(f"    PLOT SAVED {fname}")
plt.close()


# Compute CLs
print("  4. Power spectrum computation")
print("  4A. Masked cmbEB without purification")
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

print("  4B. Masked cmbEB with purification")
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

print("  4C. Masked cmbB (without purification)")
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


if not ignore_filtering:
    print("  4D. Filtered cmbEB without purification")
    fname = f"{out_dir}/cls_filtered_nopure.npz"
    if os.path.isfile(fname) and not overwrite:
        cls_filtered_nopure = np.load(fname, allow_pickle=True)['cls']
    else:
        cls_filtered_nopure = []
        for i in range(nsims_cmb):
            if i % 10 == 0:
                print("   ", i)
            mp = load_cmb_sim(i, filtered=True)
            if i == 0:
                mu.plot_map(
                    mp,
                    file_name=f"{plot_dir}/sim_cmb_filt_{i:04d}",
                    pix_type=pix_type
                )
                mu.plot_map(
                    get_masked_map(
                        mask, mp, nmt_purify=False, pix_type=pix_type
                    ),
                    file_name=f"{plot_dir}/sim_cmb_filt_masked_{i:04d}",
                    pix_type=pix_type
                )
            cl = compute_pspec(
                mp, mask, nmt_bins, transfer=None,
                wcs=wcs, nmt_purify=True
            )
            cls_filtered_nopure.append(cl)
        cls_filtered_nopure = np.array(cls_filtered_nopure)
        np.savez(fname, cls=cls_filtered_nopure)
    
    print("  4E. Filtered cmbEB without purification, TFed")
    fname = f"{out_dir}/cls_filtered_nopure_tfed.npz"
    if os.path.isfile(fname) and not overwrite:
        cls_filtered_nopure_tfed = np.load(fname, allow_pickle=True)['cls']
    else:
        cls_filtered_nopure_tfed = []
        for i in range(nsims_cmb):
            if i % 10 == 0:
                print("   ", i)
            mp = load_cmb_sim(i, filtered=True)
            if i == 0:
                mu.plot_map(
                    mp,
                    file_name=f"{plot_dir}/sim_cmb_filt_{i:04d}",
                    pix_type=pix_type
                )
                mu.plot_map(
                    get_masked_map(
                        mask, mp, nmt_purify=False, pix_type=pix_type
                    ),
                    file_name=f"{plot_dir}/sim_cmb_filt_masked_{i:04d}",
                    pix_type=pix_type
                )
            cl = compute_pspec(
                mp, mask, nmt_bins, transfer=transfer_nopure["full_tf"],
                wcs=wcs, nmt_purify=False
            )
            cls_filtered_nopure_tfed.append(cl)
        cls_filtered_nopure_tfed = np.array(cls_filtered_nopure_tfed)
        np.savez(fname, cls=cls_filtered_nopure_tfed)

    print("  4F. Filtered cmbEB with purification")
    fname = f"{out_dir}/cls_filtered_pure.npz"
    if os.path.isfile(fname) and not overwrite:
        cls_filtered_pure = np.load(fname, allow_pickle=True)['cls']
    else:
        cls_filtered_pure = []
        for i in range(nsims_cmb):
            if i % 10 == 0:
                print("   ", i)
            mp = load_cmb_sim(i, filtered=True)
            cl = compute_pspec(
                mp, mask, nmt_bins, wcs=wcs, transfer=None, nmt_purify=True
            )
            if i == 0:
                mu.plot_map(
                    mp, file_name=f"{plot_dir}/sim_cmb_filt_purify_{i:04d}",
                    pix_type="car"
                )
            cls_filtered_pure.append(cl)
        cls_filtered_pure = np.array(cls_filtered_pure)
        np.savez(fname, cls=cls_filtered_pure)
    
    print("  4G. Fitered cmbEB with purification, TFed")
    fname = f"{out_dir}/cls_filtered_pure_tfed.npz"
    if os.path.isfile(fname) and not overwrite:
        cls_filtered_pure_tfed = np.load(fname, allow_pickle=True)['cls']
    else:
        cls_filtered_pure_tfed = []
        for i in range(nsims_cmb):
            if i % 10 == 0:
                print("   ", i)
            mp = load_cmb_sim(i, filtered=True)
            cl = compute_pspec(
                mp, mask, nmt_bins, wcs=wcs, transfer=transfer_pure["full_tf"],
                nmt_purify=True
            )

            if i == 0:
                mu.plot_map(
                    mp, file_name=f"{plot_dir}/sim_cmb_filt_purify_{i:04d}",
                    pix_type="car"
                )
            cls_filtered_pure_tfed.append(cl)
        cls_filtered_pure_tfed = np.array(cls_filtered_pure_tfed)
        np.savez(fname, cls=cls_filtered_pure_tfed)

    print("  4H. Filtered cmbB (without purification)")
    fname = f"{out_dir}/cls_noe_filtered.npz"
    if os.path.isfile(fname) and not overwrite:
        cls_noe_filtered = np.load(fname, allow_pickle=True)['cls']
    else:
        cls_noe_filtered = []
        for i in range(nsims_cmb):
            if i % 10 == 0:
                print("   ", i)
            mp = load_cmb_sim(i, filtered=True, pols_keep="B")
            cl = compute_pspec(
                mp, mask, nmt_bins, wcs=wcs, transfer=None
            )
            cls_noe_filtered.append(cl)
        cls_noe_filtered = np.array(cls_noe_filtered)
        np.savez(fname, cls=cls_noe_filtered)

    print("  4J. Filtered cmbB (without purification), TFed")
    fname = f"{out_dir}/cls_noe_filtered_tfed.npz"
    if os.path.isfile(fname) and not overwrite:
        cls_noe_filtered_tfed = np.load(fname, allow_pickle=True)['cls']
    else:
        cls_noe_filtered_tfed = []
        for i in range(nsims_cmb):
            if i % 10 == 0:
                print("   ", i)
            mp = load_cmb_sim(i, filtered=True, pols_keep="B")
            cl = compute_pspec(
                mp, mask, nmt_bins, wcs=wcs,
                transfer=transfer_nopure["full_tf"]
            )
            cls_noe_filtered_tfed.append(cl)
        cls_noe_filtered_tfed = np.array(cls_noe_filtered_tfed)
        np.savez(fname, cls=cls_noe_filtered_tfed)

    print("  4K. Filtered with purification and deprojection")
    fname = f"{out_dir}/cls_filtered_pure_dep_nsims{nsims_purify}.npz"
    if os.path.isfile(fname) and not overwrite:
        cls_filtered_pure_dep = np.load(fname, allow_pickle=True)['cls']
    else:
        cls_filtered_pure_dep = []
        for i in range(nsims_cmb):
            if i % 10 == 0:
                print("   ", i)
            mp = load_cmb_sim(i, filtered=True)
            mp_masked = get_masked_map(mask, mp, nmt_purify=True, pix_type=pix_type)
            mp_masked_dep, mp_template = deproject_many(mp_masked, mp_sims, mat)
            if i == 0:
                mu.plot_map(
                    mp_template,
                    file_name=f"{plot_dir}/sim_cmb_filt_template_{i:04d}",
                    pix_type=pix_type
                )
                mu.plot_map(
                    mp_masked,
                    file_name=f"{plot_dir}/sim_cmb_filt_purify_{i:04d}",
                    pix_type=pix_type
                )
                mu.plot_map(
                    mp_masked_dep,
                    file_name=f"{plot_dir}/sim_cmb_filt_purify_dep_{i:04d}",
                    pix_type=pix_type
                )
            cl = compute_pspec(
                mp_masked_dep, mask, nmt_bins,
                transfer=None,
                nmt_purify=False, nmt_purify_mcm=True,
                wcs=wcs
            )
            cls_filtered_pure_dep.append(cl)

        cls_filtered_pure_dep = np.array(cls_filtered_pure_dep)
        np.savez(fname, cls=cls_filtered_pure_dep)

    print("  4L. Filtered with purification and deprojection, TFed")
    fname = f"{out_dir}/cls_filtered_pure_dep_tfed_nsims{nsims_purify}.npz"
    if os.path.isfile(fname) and not overwrite:
        cls_filtered_pure_dep_tfed = np.load(fname, allow_pickle=True)['cls']
    else:
        cls_filtered_pure_dep_tfed = []
        for i in range(nsims_cmb):
            if i % 10 == 0:
                print("   ", i)
            mp = load_cmb_sim(i, filtered=True)
            mp_masked = get_masked_map(mask, mp, nmt_purify=True, pix_type=pix_type)
            mp_masked_dep, mp_template = deproject_many(mp_masked, mp_sims, mat)
            cl = compute_pspec(
                mp_masked_dep, mask, nmt_bins,
                transfer=transfer_pure_dep["full_tf"],
                nmt_purify=False, nmt_purify_mcm=True,
                wcs=wcs, masked_on_input=True
            )
            cls_filtered_pure_dep_tfed.append(cl)

        cls_filtered_pure_dep_tfed = np.array(cls_filtered_pure_dep_tfed)
        np.savez(fname, cls=cls_filtered_pure_dep_tfed)


# Plotting
print("  5. Plotting")
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
plt.plot(clth["BB"][:lmax+1], 'r--', alpha=0.5, label="Theory")

plt.xlim([2, lmax_plot])
plt.ylim((1e-8, 1e-2))
plt.xlabel(r"$\ell$", fontsize=14)
plt.ylabel(r"$C_\ell^{BB}$", fontsize=14)
plt.yscale('log')
plt.legend()
print(f"  PLOT SAVED {plot_dir}/cl_masked.pdf")
plt.savefig(f"{plot_dir}/cl_masked.pdf")
plt.close()


thbb = nmt_bins.bin_cell(clth["BB"][:nmt_bins.lmax+1])
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


if not ignore_filtering:
    for pols in ["EE", "EB", "BB"]:
        plt.title('Cl, w/ filtering, TFed')

        y = np.mean(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)
        plt.plot(leff, y, color='b', ls="-", label="Filtered CMB, B only")
        plt.fill_between(leff, y-yerr, y+yerr, color='b', alpha=0.2)

        y = np.mean(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)
        plt.plot(leff, y, color='k', ls="-", label="Filtered CMB, no purification")
        plt.fill_between(leff, y-yerr, y+yerr, color='k', alpha=0.2)

        y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)
        plt.plot(leff, y, color='y', ls="-", label="Filtered CMB, purified")
        plt.fill_between(leff, y-yerr, y+yerr, color='y', alpha=0.2)

        y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_dep_tfed]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_dep_tfed]), axis=0)
        plt.plot(leff, y, color='g', ls="-", label="fCMB, purified & deproj.")
        plt.fill_between(leff, y-yerr, y+yerr, color='g', alpha=0.2)

        plt.fill_between(leff, y-yerr, y+yerr, color='y', alpha=0.2)
        plt.plot(clth["BB"][:lmax+1], 'r--', alpha=0.5, label="Theory")

        plt.xlim([2, lmax_plot])
        plt.ylim((1e-8, 1e-2))
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(fr"$C_\ell^{{{pols}}}$", fontsize=14)
        plt.yscale('log')
        plt.legend()
        print(f"    PLOT SAVED {plot_dir}/cl_{pols}_filtered_tfed.pdf")
        plt.savefig(f"{plot_dir}/cl_{pols}_filtered_tfed.pdf")
        plt.close()

    for pols in ["EE", "EB", "BB"]:
        thbb = nmt_bins.bin_cell(clth[pols][:nmt_bins.lmax+1])
        y = np.mean(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)
        plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
                'b-', alpha=0.5, label="Filtered CMB, B only")

        thbb = clth_fil_nopure[pols]
        y = np.mean(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)
        plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
                'k-', alpha=0.5, label="Filtered CMB, no purification")

        thbb = clth_fil_pure[pols]
        y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)
        plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
                'y-', alpha=0.5, label="Filtered CMB, purified")

        y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_dep_tfed]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_dep_tfed]), axis=0)
        plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
                'g-', alpha=0.5,
                label=f"fCMB, purif. & deproj. (N={nsims_purify})")
        plt.axhline(0, color="r", ls="--")

        plt.xlim([2, lmax_plot])
        plt.ylim((-20, 20))
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(fr"$(\hat{{C}}_\ell^{{{pols}}} - C_\ell^{{{pols},\, th}})/(\sigma(C_\ell^{{{pols}}})/\sqrt{{N_{{\rm sims}}}})$",  # noqa
                fontsize=14)
        plt.legend()
        print(f"    PLOT SAVED {plot_dir}/cl_{pols}_bias_filtered_tfed.pdf")
        plt.savefig(f"{plot_dir}/cl_{pols}_bias_filtered_tfed.pdf")
        plt.close()

    for pols in ["EE", "EB", "BB"]:
        yerr_ref = np.std(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)
        plt.plot(leff, sigma_masked, 'b-', alpha=0.5, label="Masked CMB, purified")
        plt.plot(leff, yerr/yerr_ref, 'k-', alpha=0.5,
                label="Filtered CMB, no purification")
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)
        plt.plot(leff, yerr/yerr_ref, 'y-', alpha=0.5, label="Filtered CMB, purified")
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_dep_tfed]), axis=0)
        plt.plot(leff, yerr/yerr_ref, 'g-', alpha=0.5,
                label="fCMB, purified & deproj.")

        plt.axhline(1, color="k")

        plt.xlim([2, lmax_plot])
        plt.ylim((0.5, 1000))
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(fr"$\sigma(C_\ell^{{{pols}, X}})/\sigma(C_\ell^{{{pols}}},\,\rm{{B only}})$",
                   fontsize=14)
        plt.yscale('log')
        plt.legend()
        print(f"    PLOT SAVED {plot_dir}/cl_{pols}_error_filtered_tfed.pdf")
        plt.savefig(f"{plot_dir}/cl_{pols}_error_filtered_tfed.pdf")
        plt.close()


    plt.title('Cl, w/ filtering not TFed')
    for pols in ["EE", "EB", "BB"]:
        y = np.mean(np.array([cl[pols] for cl in cls_noe_filtered]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_noe_filtered]), axis=0)
        plt.plot(leff, y, color='b', ls="-", label="Filtered CMB, B only")
        plt.fill_between(leff, y-yerr, y+yerr, color='b', alpha=0.2)

        y = np.mean(np.array([cl[pols] for cl in cls_filtered_nopure]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure]), axis=0)
        plt.plot(leff, y, color='k', ls="-", label="Filtered CMB, no purification")
        plt.fill_between(leff, y-yerr, y+yerr, color='k', alpha=0.2)

        y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure]), axis=0)
        plt.plot(leff, y, color='y', ls="-", label="Filtered CMB, purified")
        plt.fill_between(leff, y-yerr, y+yerr, color='y', alpha=0.2)

        y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_dep]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_dep]), axis=0)
        plt.plot(leff, y, color='g', ls="-",
                label=f"fCMB, purified & deproj. (N={nsims_purify})")
        plt.fill_between(leff, y-yerr, y+yerr, color='g', alpha=0.2)

        plt.fill_between(leff, y-yerr, y+yerr, color='y', alpha=0.2)
        plt.plot(clth[pols][:lmax+1], 'r--', alpha=0.5, label="Theory")

        plt.xlim([2, lmax_plot])
        plt.ylim((1e-8, 1e-2))
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(fr"$C_\ell^{{{pols}}}$", fontsize=14)
        plt.yscale('log')
        plt.legend()
        print(f"    PLOT SAVED {plot_dir}/cl_{pols}_filtered.pdf")
        plt.savefig(f"{plot_dir}/cl_{pols}_filtered.pdf")
        plt.close()

    for pols in ["EE", "EB", "BB"]:
        thbb = nmt_bins.bin_cell(clth[pols][:nmt_bins.lmax+1])
        y = np.mean(np.array([cl[pols] for cl in cls_noe_filtered]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_noe_filtered]), axis=0)
        plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
                'b-', alpha=0.5, label="Filtered CMB, B only")

        thbb = clth_fil_nopure[pols]
        y = np.mean(np.array([cl[pols] for cl in cls_filtered_nopure]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure]), axis=0)
        plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
                'k-', alpha=0.5, label="Filtered CMB, no purification")

        thbb = clth_fil_pure[pols]
        y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure]), axis=0)
        plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
                'y-', alpha=0.5, label="Filtered CMB, purified")

        y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_dep]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_dep]), axis=0)
        plt.plot(leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)),
                'g-', alpha=0.5,
                label=f"fCMB, purified & deproj. (N={nsims_purify})")
        plt.axhline(0, color="r", ls="--")

        plt.xlim([2, lmax_plot])
        plt.ylim((-20, 20))
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(fr"$(\hat{{C}}_\ell^{{{pols}}} - C_\ell^{{{pols},\, th}})/(\sigma(C_\ell^{{{pols}}})/\sqrt{{N_{{\rm sims}}}})$",  # noqa
                   fontsize=14)
        plt.legend()
        print(f"    PLOT SAVED {plot_dir}/cl_{pols}_bias_filtered.pdf")
        plt.savefig(f"{plot_dir}/cl_{pols}_bias_filtered.pdf")
        plt.close()

    for pols in ["EE", "EB", "BB"]:
        yerr_ref = np.std(np.array([cl[pols] for cl in cls_noe_filtered]), axis=0)
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure]), axis=0)
        plt.plot(leff, sigma_masked, 'b-', alpha=0.5, label="Masked CMB, purified")
        plt.plot(leff, yerr/yerr_ref, 'k-', alpha=0.5,
                label="Filtered CMB, no purification")
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure]), axis=0)
        plt.plot(leff, yerr/yerr_ref, 'y-', alpha=0.5, label="Filtered CMB, purified")
        yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_dep]), axis=0)
        plt.plot(leff, yerr/yerr_ref, 'g-', alpha=0.5,
                label=f"fCMB, purified & deproj. (N={nsims_purify})")

        plt.axhline(1, color="k")

        plt.xlim([2, lmax_plot])
        plt.ylim((0.5, 1000))
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(fr"$\sigma(C_\ell^{{{pols}, X}})/\sigma(C_\ell^{{{pols}}},\,\rm{{B only}})$",
                fontsize=14)
        plt.yscale('log')
        plt.legend()
        print(f"    PLOT SAVED {plot_dir}/cl_{pols}_error_filtered.pdf")
        plt.savefig(f"{plot_dir}/cl_{pols}_error_filtered.pdf")
        plt.close()
