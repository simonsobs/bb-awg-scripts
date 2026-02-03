import camb
import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
from pixell import enmap, curvedsky, enplot
from scipy.linalg import pinvh
from itertools import product


# # Changelog:
# * 2025/12/20: deproject only the QU part
# * 2026/01/17: fix bug related to field pairs ordering in TF calculation


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
        bl = np.exp(-0.5*lth*(lth+1)*np.radians(beam_fwhm/60.)**2)
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
    if "EE_to_EE" not in tf_dict:
        cases = list(tf_dict.keys())

    for id1, f1 in enumerate(field_pairs):
        for id2, f2 in enumerate(field_pairs):
            ax = plt.subplot(grid[id1, id2])
            expected = 1. if f1 == f2 else 0.
            ylims = [0, 1.05] if f1 == f2 else [-0.01, 0.01]

            ax.axhline(expected, color="k", ls="--", zorder=6)
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
    print(f"    PLOT SAVED {file_name}")


def get_inv_coupling(coupling_fname, mask, nmt_bins,
                     transfer=None, nmt_purify=False,
                     return_bp_win=False, wcs=None, lmax_mask=None,
                     overwrite=False):
    """
    """
    if os.path.isfile(coupling_fname) and not overwrite:
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

    # We (one-sided-) bin the MCM
    bmcm = np.einsum('ij,kjlm->kilm', binner, mcm)
    bmcmb = np.transpose(
        np.array([
            np.sum(bmcm[:, :, :, nmt_bins.get_ell_list(i)],
                   axis=-1)
            for i in range(nbins)
        ]), axes=[1, 2, 3, 0]
    )
    if transfer is not None:
        bmcm = np.einsum('ijk,jklm->iklm', transfer, bmcm)
        bmcmb = np.einsum('ijk,jklm->iklm', transfer, bmcmb)

    inv_coupling = np.linalg.inv(bmcmb.reshape((9*nbins, 9*nbins)))
    bp_win = np.dot(inv_coupling, bmcm.reshape([9*nbins, 9*nl])).reshape([9, nbins, 9, nl])  # noqa: E501
    np.savez(coupling_fname, inv_coupling=inv_coupling, bp_win=bp_win)

    if return_bp_win:
        return inv_coupling, bp_win
    return inv_coupling


def read_map(map_file,
             pix_type='car',
             fields_hp=None,
             convert_K_to_muK=False,
             geometry=None,
             car_template=None):
    """
    Read a map from a file, regardless of the pixelization type.

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
    car_template: str
        Path to CAR geometry template.

    Returns
    -------
    map_out : np.ndarray
        Loaded map.
    """
    conv = 1
    if convert_K_to_muK:
        conv = 1.e6
    if not (pix_type in ['hp', 'car']):
        raise ValueError(f"Unknown pixelisation type {pix_type}.")
    if pix_type == 'hp':
        kwargs = {"field": fields_hp} if fields_hp is not None else {}
        m = hp.read_map(map_file, **kwargs)
    else:
        if geometry is None:
            geometry = enmap.read_map_geometry(car_template)
        m = enmap.read_map(map_file, geometry=geometry)

    return conv*m


def get_coupled_pseudo_cls(fields1, fields2, nmt_binning):
    """
    Compute the binned coupled pseudo-C_ell estimates from two
    (spin-0 or spin-2) NaMaster fields and a multipole binning scheme.
    Parameters
    ----------
    fields1, fields2 : NmtField
        Spin-0 or spin-2 fields to correlate.
    nmt_binning : NmtBin
        Multipole binning scheme.
    """
    spins = list(fields1.keys())

    pcls = {}
    for spin1 in spins:
        for spin2 in spins:

            f1 = fields1[spin1]
            f2 = fields2[spin2]

            coupled_cell = nmt.compute_coupled_cell(f1, f2)
            coupled_cell = coupled_cell[:, :nmt_binning.lmax+1]

            pcls[f"{spin1}x{spin2}"] = nmt_binning.bin_cell(coupled_cell)
    return pcls


def decouple_pseudo_cls(coupled_pseudo_cells, coupling_inv):
    """
    Decouples the coupled pseudo-C_ell estimators computed between two fields
    of spin 0 or 2. Returns decoupled binned power spectra labeled by field
    pairs (e.g. 'TT', 'TE', 'EE', 'EB', 'BB' etc.).
    Parameters
    ----------
    coupled_pseudo_cells : dict with keys f"spin{s1}xspin{s2}",
        items array-like. Coupled pseudo-C_ell estimators.
    coupling_inv : array-like
        Inverse binned bandpower coupling matrix.
    """
    decoupled_pcls = {}
    stacked_pcls = np.concatenate(
        np.vstack([
            coupled_pseudo_cells["spin0xspin0"],
            coupled_pseudo_cells["spin0xspin2"],
            coupled_pseudo_cells["spin2xspin0"],
            coupled_pseudo_cells["spin2xspin2"]
        ])
    )
    decoupled_pcls_vec = coupling_inv @ stacked_pcls

    field_pairs = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    nbins = coupled_pseudo_cells["spin0xspin0"].shape[-1]
    for i, fp in enumerate(field_pairs):
        decoupled_pcls[fp] = decoupled_pcls_vec[i*nbins:(i+1)*nbins]
    return decoupled_pcls


def map2alm(map, lmax, pix_type="car"):
    """
    """
    if not (pix_type in ['hp', 'car']):
        raise ValueError(f"Unknown pixelisation type {pix_type}.")

    if isinstance(map, str):
        map = read_map(map, pix_type=pix_type)

    if pix_type == "hp":
        return hp.map2alm(map, lmax=lmax)
    else:
        return curvedsky.map2alm(map, lmax=lmax)


def alm2map(alm, pix_type="car", nside=None, car_map_template=None):
    """
    """
    if not (pix_type in ['hp', 'car']):
        raise ValueError(f"Unknown pixelisation type {pix_type}.")
    if isinstance(alm, list):
        alm = np.array(alm, dtype=np.complex128)

    if pix_type == "hp":
        assert nside is not None, "nside is required"
        return hp.alm2map(alm, nside=nside)
    else:
        if isinstance(car_map_template, str):
            shape, wcs = enmap.read_map_geometry(car_map_template)
        else:
            shape, wcs = car_map_template.geometry
        map = enmap.zeros((3,) + shape, wcs)
        return curvedsky.alm2map(alm, map)


def lmax_from_map(map, pix_type="car"):
    """
    Determine the maximum multipole from a map and its
    pixellization type.

    Parameters
    ----------
    map : str or np.ndarray or enmap.ndmap
        Input filename or map.
    pix_type : str, optional
        Pixellization type.

    Returns
    -------
    int
        Maximum multipole.
    """
    if not (pix_type in ['hp', 'car']):
        raise ValueError(f"Unknown pixelisation type {pix_type}.")

    if isinstance(map, str):
        if pix_type == "car":
            _, wcs = enmap.read_map_geometry(map)
            lmax = int(np.pi/np.deg2rad(np.min(np.abs(wcs.wcs.cdelt))))
            return lmax
        else:
            map = read_map(map)
    if pix_type == "hp":
        nside = hp.npix2nside(map.shape[-1])
        return 3 * nside - 1
    else:
        _, wcs = map.geometry
        lmax = int(np.pi/np.deg2rad(np.min(np.abs(wcs.wcs.cdelt))))
        return lmax


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
    mp_h = mp[-2:].flatten()
    # [ndeproj, 2*npix]
    ndeproj = len(simvec)  # ndeproj is the number of sims to deproject
    simvec_h = simvec[:, -2:, :].reshape([ndeproj, 2*npix])

    # [ndeproj, ndeproj]
    Nij = mat[:ndeproj][:, :ndeproj]
    # [ndeproj]
    prods = np.sum(simvec_h*mp_h[None, :], axis=-1)

    # [ndeproj]
    Niprod = np.dot(pinvh(Nij, rtol=1E-3), prods)
    # Niprod = np.linalg.solve(Nij, prods)
    # [ndeproj, 2*npix] * [ndeproj] = [2*npix]
    mcont = np.dot(Niprod, simvec_h)
    # [3, npix]
    mp_deproject = np.array([mp[0]] + list(mcont.reshape(mp[-2:].shape)))
    mp_deprojected = np.array(
        [mp[0]] + list((mp_h - mcont).reshape(mp[-2:].shape))
    )
    if wcs is not None:
        mp_deproject = enmap.ndmap(mp_deproject, wcs=wcs)
        mp_deprojected = enmap.ndmap(mp_deprojected, wcs=wcs)

    return mp_deprojected, mp_deproject


def _plot_map_hp(map, lims=None, file_name=None, title=None):
    """
    Hidden function to plot HEALPIX maps and either show it
    or save it to a file.


    Parameters
    ----------
    map : np.ndarray
        Input map.
    lims : list, optional
        Color scale limits.
        If map is a single component, lims is a list [min, max].
        If map is a 3-component map, lims is a list of 2-element lists.
    file_name : str, optional
        Output file name.
    title : str, optional
        Plot title.
    """
    ncomp = map.shape[0] if len(map.shape) == 2 else 1
    cmap = "YlOrRd" if ncomp == 1 else "RdYlBu_r"
    if lims is None:
        range_args = [{} for i in range(ncomp)]

    if ncomp == 1 and lims is not None:
        range_args = [{
            "min": lims[0],
            "max": lims[1]
        }]
    if ncomp != 1 and lims is not None:
        range_args = [
            {
                "min": lims[i][0],
                "max": lims[i][1]
            } for i in lims(3)
        ]
    for i in range(ncomp):
        if ncomp != 1:
            f = "TQU"[i]
        hp.mollview(
            np.atleast_2d(map)[i],
            cmap=cmap,
            title=title,
            **range_args[i],
            cbar=True
        )
        if file_name:
            if ncomp == 1:
                plt.savefig(f"{file_name}.png", bbox_inches="tight")
            else:
                plt.savefig(f"{file_name}_{f}.png", bbox_inches="tight")
        else:
            plt.show()
        plt.close()


def _plot_map_car(map, lims=None, file_name=None):
    """
    Hidden function to plot CAR maps and either show it
    or save it to a file.

    Parameters
    ----------
    map : np.ndarray
        Input map.
    lims : list, optional
        Color scale limits.
        If map is a single component, lims is a list [min, max].
        If map is a 3-component map, lims is a list of 2-element lists.
    file_name : str, optional
        Output file name.
    """
    ncomp = map.shape[0] if len(map.shape) == 3 else 1

    if lims is None:
        range_args = {}

    if ncomp == 1 and lims is not None:
        range_args = {
            "min": lims[0],
            "max": lims[1]
        }
    if ncomp == 3 and lims is not None:
        range_args = {
            "min": [lims[i][0] for i in range(ncomp)],
            "max": [lims[i][1] for i in range(ncomp)]
        }

    plot = enplot.plot(
         map,
         colorbar=True,
         ticks=10,
         **range_args
    )
    for i in range(ncomp):
        suffix = ""
        if ncomp != 1:
            suffix = f"_{'TQU'[i]}"

        if file_name:
            enplot.write(
                f"{file_name}{suffix}.png",
                plot[i]
            )
            print(f"     PLOT {file_name}{suffix}.png")
        else:
            enplot.show(plot[i])


def plot_map(map, file_name=None, lims=None, title=None, pix_type="car"):
    """
    Plot a map regardless of the pixelization type.

    Parameters
    ----------
    map : np.ndarray
        Input map.
    file_name : str, optional
        Output file name.
    lims : list, optional
        Color scale limits.
        If map is a single component, lims is a list [min, max].
        If map is a 3-component map, lims is a list of 2-element lists.
    title : str, optional
        Plot title.
    pix_type : str, optional
        Pixellization type.
    """
    if not (pix_type in ['hp', 'car']):
        raise ValueError(f"Unknown pixelisation type {pix_type}.")

    if pix_type == "hp":
        _plot_map_hp(map, lims, file_name=file_name, title=title)
    else:
        _plot_map_car(map, lims, file_name=file_name)


def write_map(map_file, map, dtype=np.float64, pix_type='hp',
              convert_muK_to_K=False):
    """
    Write a map to a file, regardless of the pixelization type.

    Parameters
    ----------
    map_file : str
        Map file name.
    map : np.ndarray
        Map to write.
    dtype : np.dtype, optional
        Data type.
    pix_type : str, optional
        Pixellization type.
    convert_muK_to_K : bool, optional
        Convert muK to K.
    """
    if convert_muK_to_K:
        map *= 1.e-6
    if not (pix_type in ['hp', 'car']):
        raise ValueError(f"Unknown pixelisation type {pix_type}.")
    if pix_type == 'hp':
        hp.write_map(map_file, map, overwrite=True, dtype=dtype)
    else:
        enmap.write_map(map_file, map)


def get_pcls_mat_transfer(fields, nmt_binning, fields2=None):
    """
    Compute coupled binned pseudo-C_ell estimates from
    pure-E and pure-B transfer function estimation simulations,
    and cast them into matrix shape.

    Parameters
    ----------
    fields: dictionary of NmtField objects (keys "pureE", "pureB")
    nmt_binning: NmtBin object
    fields2: dict, optional
        If not None, compute the pseudo-C_ell estimators
        from the cross-correlation of the fields in `fields`
        and `fields2`.
    """
    if fields2 is None:
        fields2 = fields

    n_bins = nmt_binning.get_n_bands()
    pcls_mat = np.zeros((9, 9, n_bins))

    cases = ["pureT", "pureE", "pureB"]
    tmp_pcls = {}
    for pure_type1, pure_type2 in product(cases, cases):
        pcls = get_coupled_pseudo_cls(fields[pure_type1],
                                      fields2[pure_type2],
                                      nmt_binning)
        tmp_pcls[pure_type1, pure_type2] = {
            "TT": pcls["spin0xspin0"][0],
            "TE": pcls["spin0xspin2"][0],
            "TB": pcls["spin0xspin2"][1],
            "EE": pcls["spin2xspin2"][0],
            "EB": pcls["spin2xspin2"][1],
            "BE": pcls["spin2xspin2"][2],
            "BB": pcls["spin2xspin2"][3]
        }

    field_pairs = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    for idx, fp in enumerate(field_pairs):
        pure_type1, pure_type2 = (f"pure{fp[0]}", f"pure{fp[1]}")
        pcls_mat[idx] = np.array([
            tmp_pcls[pure_type1, pure_type2]["TT"],
            tmp_pcls[pure_type1, pure_type2]["TE"],
            tmp_pcls[pure_type1, pure_type2]["TB"],
            tmp_pcls[pure_type2, pure_type1]["TE"],
            tmp_pcls[pure_type2, pure_type1]["TB"],
            tmp_pcls[pure_type1, pure_type2]["EE"],
            tmp_pcls[pure_type1, pure_type2]["EB"],
            tmp_pcls[pure_type1, pure_type2]["BE"],
            tmp_pcls[pure_type1, pure_type2]["BB"]
        ])

    return pcls_mat


def bin_theory_cls(cls, bpwf):
    """
    """
    fields_theory = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}
    fields_all = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    nl_th = cls["TT"].shape[0]

    size, n_bins, _, nl = bpwf.shape
    assert size == 9, "Unexpected number of fields in coupling matrix"
    assert nl <= nl_th, f"Theory spectrum must contain ell up to {nl}."

    cls_dict = {}
    for fp in fields_all:
        if fp in fields_theory:
            cls_dict[fp] = cls[fp][:nl]
        else:
            cls_dict[fp] = np.zeros(nl)
    cls_vec = np.array([cls_dict[fp] for fp in fields_all])

    clb = np.dot(bpwf.reshape(size*n_bins, size*nl), cls_vec.reshape(size*nl))
    clb = clb.reshape(size, n_bins)

    return {fp: clb[ifp] for ifp, fp in enumerate(fields_all)}


def get_transfer_with_error(mean_pcls_mat_filt,
                            mean_pcls_mat_unfilt,
                            pcls_mat_filt):
    """
    from SOOPERCOOL
    """
    cct_inv = np.transpose(
        np.linalg.inv(
            np.transpose(
                np.einsum('jil,jkl->ikl',
                          mean_pcls_mat_unfilt,
                          mean_pcls_mat_unfilt),
                axes=[2, 0, 1]
            )
        ), axes=[1, 2, 0]
    )

    tf = np.einsum(
        'ijl,jkl->kil', cct_inv,
        np.einsum(
            'jil,jkl->ikl',
            mean_pcls_mat_unfilt,
            mean_pcls_mat_filt
        )
    )

    tferr = np.std(
        np.array(
            [np.einsum(
                'ijl,jkl->kil', cct_inv,
                np.einsum(
                    'jil,jkl->ikl',
                    mean_pcls_mat_unfilt,
                    clf))
                for clf in pcls_mat_filt]
        ), axis=0
    )

    return tf, tferr


def get_transfer_dict(pcls_mat_filt, pcls_mat_unfilt):
    """
    from SOOPERCOOL, modified
    """
    mean_pcls_mat_filt = np.mean(pcls_mat_filt, axis=0)
    mean_pcls_mat_unfilt = np.mean(pcls_mat_unfilt, axis=0)

    tf, tferr = get_transfer_with_error(mean_pcls_mat_filt,
                                        mean_pcls_mat_unfilt,
                                        pcls_mat_filt)
    field_pairs = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    tf_dict = {}
    for i, fp1 in enumerate(field_pairs):
        for j, fp2 in enumerate(field_pairs):
            tf_dict[f"{fp2}_to_{fp1}"] = tf[i, j]
            tf_dict[f"{fp2}_to_{fp1}_std"] = tferr[i, j]
    tf_dict["full_tf"] = tf

    return tf_dict


def plot_pcls_mat_transfer(pcls_mat_unfilt, pcls_mat_filt, lb, file_name,
                           lmax=None):
    """
    """
    field_pairs = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    plt.figure(figsize=(25, 25))
    grid = plt.GridSpec(9, 9, hspace=0.3, wspace=0.3)
    msk = np.ones_like(lb).astype(bool)
    if lmax is not None:
        msk = lb <= lmax

    for id1, f1 in enumerate(field_pairs):
        for id2, f2 in enumerate(field_pairs):
            ax = plt.subplot(grid[id1, id2])
            ax.text(0.5, 0.5, f"{id1} {id2}",
                    transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            ax.set_title(f"{f1} $\\rightarrow$ {f2}", fontsize=14)
            ax.plot(lb[msk], pcls_mat_unfilt[id1, id2][msk],
                    c="navy", label="unfiltered")
            ax.plot(lb[msk], -pcls_mat_unfilt[id1, id2][msk],
                    c="navy", ls="--")
            ax.plot(lb[msk], pcls_mat_filt[id1, id2][msk],
                    c="darkorange", label="filtered")
            ax.plot(lb[msk], -pcls_mat_filt[id1, id2][msk],
                    c="darkorange", ls="--")
            ax.set_yscale("log")
            if id1 == 8:
                ax.set_xlabel(r"$\ell$", fontsize=14)
            else:
                ax.set_xticks([])
            if (id1, id2) == (0, 0):
                ax.legend(fontsize=14)

    plt.savefig(file_name, bbox_inches="tight")
    print(f"   PLOT  {file_name}")
    plt.close()
