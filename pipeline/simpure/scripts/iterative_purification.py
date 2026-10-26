import numpy as np
import matplotlib.pyplot as plt
import os
import utils as ut
from utils import MatrixPurification
from simpure import SimPure
import sys
import argparse
import yaml

# TODO: Make it an actual module
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'misc'))
)
import mpi_utils as mpi  # noqa


# Changelog:
# * 2025/12/19: added masked_on_input=True for deprojection spectra
# * 2025/12/20: made deprojection matrix with QU only
# * 2025/12/20: fixed bug in cls_filtered_nopure (purify was set to True)
# * 2026/01/17: fixed bug in get_pcls_mat_transfer (wrong field pair ordering)
# * 2026/01/25: ported functions to simpure
# * 2026/01/29: added flag tf_type to get_inv_coupling and compute_pspec
# * 2026/03/02: added MPI parallelization
# * 2026/03/03: changed output directory naming convention
# * 2026/03/04: added power-law simulation validation
# * 2026/05/05: tested power-law simulation validation
# * 2026/05/27: added matrix-based purification
# * 2026/08/24: added optimizd matrix purif. Added BBmaster obsmat at nside 64
# * 2026/09/11: added argparse with yaml file
# * 2026/09/24: load deprojection matrix from file, accept matrix_thresh as config  # noqa: E501
# * 2026/10/17: added no-filter option to deprojection and matrix purification
# * 2026/10/26: added option to compute TFs from CMB-EE-like power spectra


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def main(config):
    """
    Perform advance purification on a validation setup of CMB and power-law
    simulations.
    """
    pix_type = config["pix_type"]
    nside = config["nside"]
    beam_fwhm = config["beam_fwhm"]
    nlb = config["nlb"]
    plot_dl = config["plot_dl"]
    lmax_plot = config["lmax_plot"]
    car_template = config["car_template"]
    mask_file = config["mask_file"]

    test_label = config["test_label"]
    base_dir = config["base_dir"]
    purify_method = config["purify_method"]
    filter_setup = config["filter_setup"]

    threshold_quantile_matpure = config["threshold_quantile_matpure"]
    nsims_purify = config["nsims_purify"]
    nsims_val = config["nsims_val"]
    nsims_transfer = config["nsims_transfer"]
    add_noise = config["add_noise"]  # TODO
    nbundles_noise = config["nbundles"]  # TODO
    if "noise_level" in config:
        noise_level = config["noise_level"]
    else:
        noise_level = 7  # muKamin
    if "tf_sims_cl_shape" in config:
        tf_sims_cl_shape = config["tf_sims_cl_shape"]
    else:
        tf_sims_cl_shape = "plaw"

    overwrite = config["overwrite"]
    deproject_null = config["deproject_null"]
    ignore_filtering = config["ignore_filtering"]

    out_dir = f"{base_dir}/purification/{test_label}"
    plot_dir = f"{out_dir}/plots"
    id_sim_transfer_start = 0

    sp = SimPure(pix_type,
                 base_dir,
                 out_dir,
                 filter_setup,
                 mask_file,
                 nside=nside,
                 car_template=car_template,
                 beam_fwhm=beam_fwhm,
                 nlb=nlb
                 )
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
    ls = np.arange(sp.lmax+1)
    cl2dl = ls*(ls+1)/2./np.pi if plot_dl else 1.
    _, cl_cmb = ut.get_theory_cls(cosmo, lmax=sp.lmax+500, beam_fwhm=beam_fwhm)
    _, cl_plaw = ut.get_plaw_cls(sp.lmax+500, beam_fwhm=beam_fwhm)
    clth_dict = {"cmb": cl_cmb, "plaw": cl_plaw}
    deproj_sims, deproj_mat, matpure = (None, None, None)

    os.makedirs(plot_dir, exist_ok=True)
    ut.plot_map(sp.mask, file_name=f"{plot_dir}/mask", pix_type=pix_type)
    ut.plot_map(sp.mask_bin, file_name=f"{plot_dir}/mask_bin",
                pix_type=pix_type)

    if not ignore_filtering:
        if purify_method == "deproj":
            if not deproject_null:
                print("  1A. Make deprojection matrix")
                sim_dir = f"{out_dir}/sims_filt"
                sim_fn = "sim_pureE_alpha2_filt_pure_Bout_{id_sim:04d}.fits"
                mat_fn = f"{out_dir}/matcorr_alpha2_filt_pure_Bout_nsims{nsims_purify}.npz"  # noqa: E501
                mat_plot_fn = f"{plot_dir}/M_deproject_eigvals_nsims{nsims_purify}.pdf"  # noqa: E501
                deproj_sims, deproj_mat = sp.make_deprojection_matrix(
                    sim_dir, sim_fn, mat_fn, nsims_purify, overwrite,
                    mat_plot_fn)
            else:
                mp = sp.load_val_sim(0, filtered=True, typ="cmb")
                deproj_sims = np.zeros((nsims_purify,) + mp.shape)
                deproj_mat = np.eye(nsims_purify)
        elif purify_method == "matrix":
            if sp.rank == 0:
                print("   1B. Prepare matrix purification")
            if filter_setup == "obsmat_polyonly_apo_nside64":
                # NOTE: eigen_spectrum_nside64 is polyonly-filtered and apo-masked  # noqa: E501
                eigspec_fn = "/pscratch/sd/k/kwolz/bbdev/matrixpure/eigen_spectrum_nside64.npz"  # noqa: E501
            elif filter_setup == "obsmat_apo_nside64":
                # NOTE: obsmat_apo_nside64 means bbmaster-filtered and apo-masked  # noqa: E501
                eigspec_fn = "/pscratch/sd/k/kwolz/bbdev/matrixpure/eigen_spectrum_apo_mask_bbmaster_nside64.npz"  # noqa: E501
            elif filter_setup is None:
                eigspec_fn = "/pscratch/sd/k/kwolz/bbdev/matrixpure/eigen_spectrum_apo_mask_nofilter_nside64.npz"  # noqa: E501

            # Plot eigenvalues
            eigspec = np.load(eigspec_fn)
            eigvals = eigspec["eigvals"]
            plt.figure()
            filter_label = "NOFILTER" if filter_setup is None else filter_setup
            plt.plot(eigvals,
                     label=filter_label+"\n thresh_hi "+f"{threshold_quantile_matpure}")  # noqa: E501
            plt.axvline(0.25*len(eigvals), c="k")
            plt.axvline(threshold_quantile_matpure*len(eigvals), c="k")
            plt.yscale("log")
            plt.legend()
            plt.xlabel("Rank")
            plt.ylabel("Eigenvalue")
            plt.savefig(f"{plot_dir}/M_matrix_eigvals_thresh{threshold_quantile_matpure}.pdf")  # noqa: E501
            print(f"{plot_dir}/M_matrix_eigvals_thresh{threshold_quantile_matpure}.pdf")  # noqa: E501

            # Plot eigenvectors
            if sp.pix_type == "hp":
                import healpy as hp

                cmap = "RdYlBu_r"
                norm_eigvecs = eigspec['norm_eigvecs']
                vec_pureb = norm_eigvecs[:, -25:] * eigvals[-25:]  # highest 25 pure-B eigenvectors  # noqa: E501
                vec_puree = norm_eigvecs[:, :25] * eigvals[:25]  # highest 25 pure-E eigenvectors  # noqa: E501

                msk = sp.mask_bin.astype(bool)
                maps_pureb = np.zeros((2,) + (25,) + msk.shape)
                maps_puree = np.zeros((2,) + (25,) + msk.shape)
                for i in range(25):
                    maps_pureb[:, i, msk] = vec_pureb[:, i].reshape((2, -1))
                    maps_puree[:, i, msk] = vec_puree[:, i].reshape((2, -1))

                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize
                fig = plt.figure(figsize=(15, 10))
                ext = np.max((np.fabs(np.min(maps_pureb)),
                              np.fabs(np.max(maps_pureb))))
                RA0, Dec0 = 15, -40
                lonra = [-75, 75]
                latra = [-45, 30]
                for i in range(25):
                    hp.cartview(
                        maps_pureb[0, i],
                        fig=fig.number,
                        sub=(5, 5, i + 1),   # subplot indexing starts at 1
                        title=f"PureB, Q, {i + 1}",
                        min=-ext,
                        max=ext,
                        cmap=cmap,
                        cbar=False,
                        notext=True,
                        rot=(RA0, Dec0, 0),
                        lonra=lonra,
                        latra=latra,
                        xsize=640
                    )
                cax = fig.add_axes([0.20, -0.04, 0.60, 0.02])
                sm = ScalarMappable(
                    norm=Normalize(vmin=-ext, vmax=ext),
                    cmap=cmap
                )
                sm.set_array([])

                cbar = fig.colorbar(
                    sm,
                    cax=cax,
                    orientation="horizontal"
                )
                cbar.set_label("Value")
                plt.savefig(f"{plot_dir}/pureB_Q_eigenvectors.png",
                            bbox_inches="tight")
                print(f"{plot_dir}/pureB_Q_eigenvectors.png")

            matpure = MatrixPurification(sp.mask,
                                         eigspec_fn,
                                         thresh_lo=0.25,
                                         thresh_hi=threshold_quantile_matpure)
        sp.comm.barrier()

        print("  2. Transfer function computation")
        local_task_ids = mpi.distribute_tasks(sp.size, sp.rank, 5, id_start=0)

        for id in local_task_ids:
            kwargs = {"deproj_sims": deproj_sims,
                      "deproj_mat": deproj_mat,
                      "matpure": matpure,
                      "cl_shape": tf_sims_cl_shape}
            if id == 0:
                print("  2A. TF sims unfiltered w/o purification")
                kwargs.update({"filtered": False, "purified": False, "method": None})  # noqa: E501
                cls_tf_unfiltered_nopure = sp.get_tf_spectra(
                    out_dir, nsims_transfer, overwrite=overwrite,
                    id_sim_start=id_sim_transfer_start, **kwargs)
            if id == 1:
                print("  2B. TF sims unfiltered w/ purification")
                kwargs.update({"filtered": False, "purified": True, "method": None})  # noqa: E501
                cls_tf_unfiltered_pure = sp.get_tf_spectra(
                    out_dir, nsims_transfer, overwrite=overwrite,
                    id_sim_start=id_sim_transfer_start, **kwargs)
            if id == 2:
                print("  2C. TF sims filtered w/o purification")
                kwargs.update({"filtered": True, "purified": False, "method": None})  # noqa: E501
                cls_tf_filtered_nopure = sp.get_tf_spectra(
                    out_dir, nsims_transfer, overwrite=overwrite,
                    id_sim_start=id_sim_transfer_start, **kwargs)
            if id == 3:
                print("  2D. TF sims filtered w/ purification")
                kwargs.update({"filtered": True, "purified": True, "method": None})  # noqa: E501
                cls_tf_filtered_pure = sp.get_tf_spectra(
                    out_dir, nsims_transfer, overwrite=overwrite,
                    id_sim_start=id_sim_transfer_start, **kwargs)
            if id == 4:
                print("  2E. TF sims filtered w/ advanced purification")
                kwargs.update({"filtered": True, "purified": True, "method": purify_method})  # noqa: E501
                cls_tf_filtered_pure_adv = sp.get_tf_spectra(
                    out_dir, nsims_transfer, overwrite=overwrite,
                    id_sim_start=id_sim_transfer_start, **kwargs
                )

        sp.comm.barrier()
        cls_tf_unfiltered_nopure = np.load(
            f"{out_dir}/cls_tf_unfiltered_nopure_nsims{nsims_transfer}.npz",
            allow_pickle=True
        )["cls"]
        cls_tf_unfiltered_pure = np.load(
            f"{out_dir}/cls_tf_unfiltered_pure_nsims{nsims_transfer}.npz",
            allow_pickle=True
        )["cls"]
        cls_tf_filtered_nopure = np.load(
            f"{out_dir}/cls_tf_filtered_nopure_nsims{nsims_transfer}.npz",
            allow_pickle=True
        )["cls"]
        cls_tf_filtered_pure = np.load(
            f"{out_dir}/cls_tf_filtered_pure_nsims{nsims_transfer}.npz",
            allow_pickle=True
        )["cls"]
        cls_tf_filtered_pure_adv = np.load(
            f"{out_dir}/cls_tf_filtered_pure_{purify_method}_nsims{nsims_transfer}.npz",  # noqa: E501
            allow_pickle=True
        )["cls"]

        local_task_ids = mpi.distribute_tasks(sp.size, sp.rank, 3, id_start=0)
        for id in local_task_ids:
            if id == 0:
                print("  2F. TF w/ purification")
                transfer_pure = ut.get_transfer_dict(cls_tf_filtered_pure,
                                                     cls_tf_unfiltered_pure)
                np.savez(f"{out_dir}/transfer_pure.npz", **transfer_pure)
            if id == 1:
                print("  2G. TF w/o purification")
                transfer_nopure = ut.get_transfer_dict(
                    cls_tf_filtered_nopure, cls_tf_unfiltered_nopure)
                np.savez(f"{out_dir}/transfer_nopure.npz", **transfer_nopure)
            if id == 2:
                print("  2H. TF with advanced purification")
                if purify_method == "matrix":
                    # cls_tf_comp = cls_tf_unfiltered_nopure
                    cls_tf_comp = cls_tf_unfiltered_pure  # New method: pure
                else:
                    cls_tf_comp = cls_tf_unfiltered_pure
                transfer_pure_adv = ut.get_transfer_dict(
                    cls_tf_filtered_pure_adv, cls_tf_comp)
                np.savez(f"{out_dir}/transfer_pure_{purify_method}.npz",
                         **transfer_pure_adv)
        sp.comm.barrier()

        transfer_pure = np.load(f"{out_dir}/transfer_pure.npz", allow_pickle=True)  # noqa: E501
        transfer_nopure = np.load(f"{out_dir}/transfer_nopure.npz", allow_pickle=True)  # noqa: E501
        transfer_pure_adv = np.load(f"{out_dir}/transfer_pure_{purify_method}.npz", allow_pickle=True)  # noqa: E501

        if sp.rank == 0:
            field_pairs = ["EE", "EB", "BE", "BB"]
            file_name = f"{plot_dir}/transfer_function.pdf"  # noqa: E501
            tf_dict = {
                "nopure": transfer_nopure, "pure": transfer_pure,
                f"pure_{purify_method}": transfer_pure_adv
            }
            ut.plot_transfer_function(sp.leff, tf_dict, 0, lmax_plot,
                                      field_pairs, file_name=file_name)
            ut.plot_transfer_function(
                sp.leff, tf_dict, 0, lmax_plot,
                ["TT", "TE", "TB", "ET", "BT"] + field_pairs,
                file_name=file_name.replace(".pdf", "_inclT.pdf"))
            ut.plot_transfer_function(
                sp.leff, tf_dict, 0, lmax_plot, ["BB"],
                file_name=file_name.replace(".pdf", "_BB.pdf"))

            # Precompute couplings
            coupling_cases = {"nopure", "pure", f"pure_{purify_method}"}
            for cc in coupling_cases:
                print(cc)
                tf = {"nopure": transfer_nopure["full_tf"],
                      "pure": transfer_pure["full_tf"],
                      f"pure_{purify_method}": transfer_pure_adv["full_tf"]}[cc]  # noqa: E501
                purify = {
                    "nopure": False,
                    "pure": True,
                    "pure_deproj": True,
                    "pure_matrix": True  # New method: True
                }[cc]
                _ = sp.get_inv_coupling(transfer=tf,
                                        tf_type=cc,
                                        nmt_purify=purify,
                                        overwrite=overwrite)
            for purify in [True, False]:
                _ = sp.get_inv_coupling(nmt_purify=purify, overwrite=overwrite)

    # Compute CLs
    print("  3. Power spectrum computation")
    sp.comm.barrier()
    cases = [{"noE": True, "filtered": False, "map": m, "purified": False, "method": None, "TFed": False} for m in ["cmb", "plaw"]]  # noqa: E501
    cases += [{"noE": False, "filtered": False, "map": m, "purified": True, "method": None, "TFed": False} for m in ["cmb", "plaw"]]  # noqa: E501
    cases += [{"noE": False, "filtered": False, "map": m, "purified": False, "method": None, "TFed": False} for m in ["cmb", "plaw"]]  # noqa: E501
    if not ignore_filtering:
        cases += [{"noE": True, "filtered": True, "map": m, "purified": False, "method": None, "TFed": tfed} for m in ["cmb", "plaw"] for tfed in [False, True]]  # noqa: E501
        cases += [{"noE": False, "filtered": True, "map": m, "purified": True, "method": p, "TFed": t} for m in ["cmb", "plaw"] for p in [None, purify_method] for t in [False, True]]  # noqa: E501
        cases += [{"noE": False, "filtered": True, "map": m, "purified": False, "method": None, "TFed": t} for m in ["cmb", "plaw"] for t in [False, True]]  # noqa: E501

    def case_label(case):
        plab = {None: "mask", "deproj": "sim-based", "matrix": "matrix"}[purify_method]  # noqa: E501
        lab = ""
        lab += "Filtered " if case["filtered"] else "Masked "
        lab += case["map"]
        lab += "B " if case["noE"] else "EB "
        lab += f"with {plab} purification" if case["purified"] else "without purification"  # noqa: E501
        lab += ", TFed" if case["TFed"] else ""
        return lab

    if sp.rank == 0:
        print("Coupling computation")

    nranks = len(cases)
    local_task_ids = mpi.distribute_tasks(sp.size, sp.rank, nranks, id_start=0)
    overwrite_cls = overwrite
    for id in local_task_ids:
        print(f" 3 ({id+1}/{nranks})  {case_label(cases[id])}")
        kwargs = cases[id]
        if kwargs["TFed"]:
            if kwargs["purified"] and kwargs["method"]:
                tf = transfer_pure_adv["full_tf"]
            elif kwargs["purified"] and not kwargs["method"]:
                tf = transfer_pure["full_tf"]
            else:
                tf = transfer_nopure["full_tf"]
        else:
            tf = None
        _ = sp.get_val_spectra(
            out_dir, nsims_val, overwrite=overwrite_cls, plot_dir=plot_dir,
            **kwargs, tf=tf, deproj_sims=deproj_sims, deproj_mat=deproj_mat,
            matpure=matpure, add_noise=add_noise,
            nbundles_noise=nbundles_noise, noise_level=noise_level
        )

    # Plotting
    sp.comm.barrier()
    if sp.rank != 0:
        return

    print("  4. Plotting")
    os.makedirs(f"{out_dir}/plots", exist_ok=True)

    _, bpw_msk_nopure = sp.get_inv_coupling(return_bp_win=True,
                                            overwrite=overwrite)
    _, bpw_msk_pure = sp.get_inv_coupling(nmt_purify=True,
                                          return_bp_win=True,
                                          overwrite=overwrite)
    if not ignore_filtering:
        _, bpw_fil_nopure = sp.get_inv_coupling(
            transfer=transfer_nopure["full_tf"],
            tf_type="nopure",
            nmt_purify=False,
            return_bp_win=True,
            overwrite=overwrite
        )
        _, bpw_fil_pure = sp.get_inv_coupling(
            transfer=transfer_pure["full_tf"],
            tf_type="pure",
            nmt_purify=True,
            return_bp_win=True,
            overwrite=overwrite
        )
        _, bpw_fil_pure_adv = sp.get_inv_coupling(
            transfer=transfer_pure_adv["full_tf"],
            tf_type=f"pure_{purify_method}",
            nmt_purify=True,  # NEW method: True
            return_bp_win=True,
            overwrite=overwrite
        )

    for mtyp in ["plaw", "cmb"]:
        clth = clth_dict[mtyp]
        clth_msk_nopure = ut.bin_theory_cls(clth_dict[mtyp], bpw_msk_nopure)
        clth_msk_pure = ut.bin_theory_cls(clth_dict[mtyp], bpw_msk_pure)
        cls_masked_nopure = np.load(f"{out_dir}/cls_{mtyp}_masked_nopure.npz", allow_pickle=True)["cls"]  # noqa: E501
        cls_masked_pure = np.load(f"{out_dir}/cls_{mtyp}_masked_pure.npz", allow_pickle=True)["cls"]  # noqa: E501
        cls_noe_masked = np.load(f"{out_dir}/cls_{mtyp}_noe_masked.npz", allow_pickle=True)["cls"]  # noqa: E501

        plt.figure()
        plt.title('Cl, no filtering')
        y = np.mean(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)*cl2dl  # noqa: E501
        yerr = np.std(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)*cl2dl  # noqa: E501
        plt.plot(sp.leff, y, color='b', ls="-", label=f"Masked {mtyp}, B only")
        plt.fill_between(sp.leff, y-yerr, y+yerr, color='b', alpha=0.2)

        y = np.mean(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)*cl2dl  # noqa: E501
        yerr = np.std(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)*cl2dl  # noqa: E501
        plt.plot(sp.leff, y, color='k', ls="-", label=f"Masked {mtyp}, no purification")  # noqa: E501
        plt.fill_between(sp.leff, y-yerr, y+yerr, color='k', alpha=0.2)

        y = np.mean(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)*cl2dl  # noqa: E501
        yerr = np.std(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)*cl2dl  # noqa: E501
        plt.plot(sp.leff, y, color='y', ls="-", label=f"Masked {mtyp}, purified")  # noqa: E501
        plt.fill_between(sp.leff, y-yerr, y+yerr, color='y', alpha=0.2)
        plt.plot(clth["BB"][:sp.lmax+1]*cl2dl, 'r--', alpha=0.5, label="Theory")  # noqa: E501

        plt.xlim([2, lmax_plot])
        ylim = (1e-6, 2e-3) if plot_dl else (1e-8, 1e-5)
        # plt.ylim(ylim)
        plt.xlabel(r"$\ell$", fontsize=14)
        ylab = r"$D_\ell^{BB}$" if plot_dl else r"$C_\ell^{BB}$"
        plt.ylabel(ylab, fontsize=14)
        plt.yscale('log')
        plt.legend()
        print(f"  PLOT SAVED {plot_dir}/cl_{mtyp}_masked_BB.pdf")
        plt.savefig(f"{plot_dir}/cl_{mtyp}_masked_BB.pdf", bbox_inches="tight")
        plt.close()

        # thbb = sp.bins.bin_cell(clth["BB"][:sp.bins.lmax+1])
        thbb = clth_msk_nopure["BB"]
        y = np.mean(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)*cl2dl  # noqa: E501
        yerr = np.std(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)*cl2dl  # noqa: E501
        plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_val)), 'b-', alpha=0.5, label=f"Masked {mtyp}, B only")  # noqa: E501

        thbb = clth_msk_nopure["BB"]
        y = np.mean(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)*cl2dl  # noqa: E501
        yerr = np.std(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)*cl2dl  # noqa: E501
        plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_val)), 'k-', alpha=0.5, label=f"Masked {mtyp}, no purification")  # noqa: E501

        thbb = clth_msk_pure["BB"]
        y = np.mean(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)*cl2dl  # noqa: E501
        yerr = np.std(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)*cl2dl  # noqa: E501
        plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_val)), 'y-', alpha=0.5, label=f"Masked {mtyp}, purified")  # noqa: E501
        plt.axhline(0, color="r", ls="--")

        plt.xlim([2, lmax_plot])
        plt.ylim((-20, 20))
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(r"$(\hat{C}_\ell^{BB} - C_\ell^{BB,\, th})/(\sigma(C_\ell^{BB})/\sqrt{N_{\rm sims}})$", fontsize=14)  # noqa: E501
        plt.legend()
        print(f"    PLOT SAVED {plot_dir}/bias_{mtyp}_masked_BB.pdf")
        plt.savefig(f"{plot_dir}/bias_{mtyp}_masked_BB.pdf", bbox_inches="tight")  # noqa: E501
        plt.close()

        yerr_ref = np.std(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)  # noqa: E501
        yerr = np.std(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)
        plt.plot(sp.leff, yerr/yerr_ref, 'b-', alpha=0.5, label=f"Masked {mtyp}, purified")  # noqa: E501
        sigma_masked = yerr/yerr_ref
        yerr = np.std(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)
        plt.plot(sp.leff, yerr/yerr_ref, 'k-', alpha=0.5, label=f"Masked {mtyp}, no purification")  # noqa: E501
        plt.axhline(1, color="k", ls="--")

        plt.xlim([2, lmax_plot])
        plt.ylim((0.5, 1000))
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(r"$\sigma(C_\ell^{BB, X})/\sigma(C_\ell^{BB,\,\rm{B only}})$", fontsize=14)  # noqa: E501
        plt.yscale('log')
        plt.legend()
        print(f"    PLOT SAVED {plot_dir}/error_{mtyp}_masked_BB.pdf")
        plt.savefig(f"{plot_dir}/error_{mtyp}_masked_BB.pdf", bbox_inches="tight")  # noqa: E501
        plt.close()

    if not ignore_filtering:
        for mtyp in ["plaw", "cmb"]:
            clth = clth_dict[mtyp]
            clth_fil_nopure = ut.bin_theory_cls(clth_dict[mtyp], bpw_fil_nopure)  # noqa: E501
            clth_fil_pure = ut.bin_theory_cls(clth_dict[mtyp], bpw_fil_pure)
            clth_fil_pure_adv = ut.bin_theory_cls(clth_dict[mtyp], bpw_fil_pure_adv)  # noqa: E501

            cls_filtered_nopure_tfed = np.load(f"{out_dir}/cls_{mtyp}_filtered_nopure_tfed.npz", allow_pickle=True)["cls"]  # noqa: E501
            cls_filtered_nopure = np.load(f"{out_dir}/cls_{mtyp}_filtered_nopure.npz", allow_pickle=True)["cls"]  # noqa: E501
            cls_filtered_pure_adv_tfed = np.load(f"{out_dir}/cls_{mtyp}_filtered_pure_{purify_method}_tfed.npz", allow_pickle=True)["cls"]  # noqa: E501
            cls_filtered_pure_adv = np.load(f"{out_dir}/cls_{mtyp}_filtered_pure_{purify_method}.npz", allow_pickle=True)["cls"]  # noqa: E501
            cls_filtered_pure_tfed = np.load(f"{out_dir}/cls_{mtyp}_filtered_pure_tfed.npz", allow_pickle=True)["cls"]  # noqa: E501
            cls_filtered_pure = np.load(f"{out_dir}/cls_{mtyp}_filtered_pure.npz", allow_pickle=True)["cls"]  # noqa: E501
            cls_noe_filtered_tfed = np.load(f"{out_dir}/cls_{mtyp}_noe_filtered_tfed.npz", allow_pickle=True)["cls"]  # noqa: E501
            cls_noe_filtered = np.load(f"{out_dir}/cls_{mtyp}_noe_filtered.npz", allow_pickle=True)["cls"]  # noqa: E501

            for pols in ["EE", "EB", "BB"]:
                plt.title('Cl w/ filtering, TFed')
                y = np.mean(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)*cl2dl  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)*cl2dl  # noqa: E501
                plt.plot(sp.leff, y, color='b', ls="-", label=f"Filtered {mtyp}, B only")  # noqa: E501
                plt.fill_between(sp.leff, y-yerr, y+yerr, color='b', alpha=0.2)

                y = np.mean(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)*cl2dl  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)*cl2dl  # noqa: E501
                plt.plot(sp.leff, y, color='k', ls="-", label=f"Filtered {mtyp}, no purification")  # noqa: E501
                plt.fill_between(sp.leff, y-yerr, y+yerr, color='k', alpha=0.2)

                y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)*cl2dl  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)*cl2dl  # noqa: E501
                plt.plot(sp.leff, y, color='y', ls="-", label=f"Filtered {mtyp}, purified")  # noqa: E501
                plt.fill_between(sp.leff, y-yerr, y+yerr, color='y', alpha=0.2)

                y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_adv_tfed]), axis=0)*cl2dl  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_adv_tfed]), axis=0)*cl2dl  # noqa: E501
                plt.plot(sp.leff, y, color='g', ls="-", label=f"f{mtyp}, {purify_method}-purified")  # noqa: E501
                plt.fill_between(sp.leff, y-yerr, y+yerr, color='g', alpha=0.2)

                plt.fill_between(sp.leff, y-yerr, y+yerr, color='y', alpha=0.2)
                plt.plot(clth[pols][:sp.lmax+1]*cl2dl, 'r--', alpha=0.5, label="Theory")  # noqa: E501
                plt.plot(sp.leff, clth_fil_pure_adv[pols], 'r-.', alpha=0.5, label="Theory pure")  # noqa: E501

                plt.xlim([2, lmax_plot])
                plt.xlabel(r"$\ell$", fontsize=14)
                ylab = fr"$D_\ell^{{{pols}}}$" if plot_dl else fr"$C_\ell^{{{pols}}}$"  # noqa: E501
                plt.ylabel(ylab, fontsize=14)
                if pols == "BB":
                    plt.yscale('log')
                    ylim = (1e-6, 2e-3) if plot_dl else (1e-8, 1e-5)
                    # plt.ylim(ylim)
                elif pols == "EE":
                    plt.yscale('log')
                    ylim = (1e-4, 1e0) if plot_dl else (1e-6, 1e-2)
                    # plt.ylim(ylim)
                plt.legend()
                print(f"    PLOT SAVED {plot_dir}/cl_{mtyp}_filtered_tfed_{pols}.pdf")  # noqa: E501
                plt.savefig(f"{plot_dir}/cl_{mtyp}_filtered_tfed_{pols}.pdf", bbox_inches="tight")  # noqa: E501
                plt.close()

            for pols in ["EE", "EB", "BB"]:
                plt.title('Cl w/ filtering, TFed')
                # thbb = sp.bins.bin_cell(clth[pols][:sp.bins.lmax+1])
                thbb = clth_fil_nopure[pols]
                y = np.mean(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)  # noqa: E501
                plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_val)), 'b-', alpha=0.5, label=f"Filtered {mtyp}, B only")  # noqa: E501

                thbb = clth_fil_nopure[pols]
                y = np.mean(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)  # noqa: E501
                plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_val)), 'k-', alpha=0.5, label=f"Filtered {mtyp}, no purification")  # noqa: E501

                thbb = clth_fil_pure[pols]
                y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)  # noqa: E501
                plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_val)), 'y-', alpha=0.5, label=f"Filtered {mtyp}, purified")  # noqa: E501

                thbb = clth_fil_pure_adv[pols]
                y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_adv_tfed]), axis=0)  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_adv_tfed]), axis=0)  # noqa: E501
                plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_val)), 'g-', alpha=0.5, label=f"f{mtyp}, {purify_method}-purified {'(N='+str(nsims_purify)+')' if purify_method == 'deproj' else ''}")  # noqa: E501
                plt.axhline(0, color="r", ls="--")

                plt.xlim([2, lmax_plot])
                plt.ylim((-20, 20))
                plt.xlabel(r"$\ell$", fontsize=14)
                plt.ylabel(fr"$(\hat{{C}}_\ell^{{{pols}}} - C_\ell^{{{pols},\, th}})/(\sigma(C_\ell^{{{pols}}})/\sqrt{{N_{{\rm sims}}}})$", fontsize=14)  # noqa: E501
                plt.legend()
                print(f"    PLOT SAVED {plot_dir}/bias_{mtyp}_filtered_tfed_{pols}.pdf")  # noqa: E501
                plt.savefig(f"{plot_dir}/bias_{mtyp}_filtered_tfed_{pols}.pdf", bbox_inches="tight")  # noqa: E501
                plt.close()

            for pols in ["EE", "EB", "BB"]:
                yerr_ref = np.std(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)  # noqa: E501
                plt.plot(sp.leff, sigma_masked, 'm-', alpha=0.5, label=f"Masked {mtyp}, purified")  # noqa: E501
                plt.plot(sp.leff, yerr/yerr_ref, 'k-', alpha=0.5, label=f"Filtered {mtyp}, no purification")  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)  # noqa: E501
                plt.plot(sp.leff, yerr/yerr_ref, 'y-', alpha=0.5, label=f"Filtered {mtyp}, purified")  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_adv_tfed]), axis=0)  # noqa: E501
                plt.plot(sp.leff, yerr/yerr_ref, 'g-', alpha=0.5, label=f"f{mtyp}, {purify_method}-purified")  # noqa: E501
                plt.axhline(1, color="b", linestyle="--", alpha=0.5, label=f"Filtered {mtyp}, B only")  # noqa: E501

                plt.xlim([2, lmax_plot])
                plt.ylim((0.5, 1000))
                plt.xlabel(r"$\ell$", fontsize=14)
                plt.ylabel(fr"$\sigma(C_\ell^{{{pols}, X}})/\sigma(C_\ell^{{{pols}}},\,\rm{{B only}})$", fontsize=14)  # noqa: E501
                plt.yscale('log')
                plt.legend()
                print(f"    PLOT SAVED {plot_dir}/error_{mtyp}_filtered_tfed_{pols}.pdf")  # noqa: E501
                plt.savefig(f"{plot_dir}/error_{mtyp}_filtered_tfed_{pols}.pdf", bbox_inches="tight")  # noqa: E501
                plt.close()

            for pols in ["EE", "EB", "BB"]:
                yerr = np.std(np.array([cl[pols] for cl in cls_noe_masked]), axis=0)*cl2dl  # noqa: E501
                plt.plot(sp.leff, yerr, 'm--', alpha=0.5, label=f"Masked {mtyp}, B only")  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)*cl2dl  # noqa: E501
                plt.plot(sp.leff, yerr, 'b--', alpha=0.5, label=f"Filtered {mtyp}, B only")  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)*cl2dl  # noqa: E501
                plt.plot(sp.leff, yerr, 'k-', alpha=0.5, label=f"Filtered {mtyp}, no purification")  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)*cl2dl  # noqa: E501
                plt.plot(sp.leff, yerr, 'y-', alpha=0.5, label=f"Filtered {mtyp}, purified")  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_adv_tfed]), axis=0)*cl2dl  # noqa: E501
                plt.plot(sp.leff, yerr, 'g-', alpha=0.5, label=f"f{mtyp}, {purify_method}-purified")  # noqa: E501

                plt.xlim([2, lmax_plot])
                plt.yscale("log")
                plt.xlabel(r"$\ell$", fontsize=14)
                lab = "D" if plot_dl else "C"
                plt.ylabel(ylab, fontsize=14)
                plt.ylabel(fr"$\sigma({{{lab}}}_\ell^{{{pols}, X}})$", fontsize=14)  # noqa: E501
                plt.yscale('log')
                plt.legend()
                print(f"    PLOT SAVED {plot_dir}/abserr_{mtyp}_filtered_tfed_{pols}.pdf")  # noqa: E501
                plt.savefig(f"{plot_dir}/abserr_{mtyp}_filtered_tfed_{pols}.pdf", bbox_inches="tight")  # noqa: E501
                plt.close()

            for pols in ["EE", "EB", "BB"]:
                plt.title('Cl w/ filtering, not TFed')
                y = np.mean(np.array([cl[pols] for cl in cls_noe_filtered]), axis=0)*cl2dl  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_noe_filtered]), axis=0)*cl2dl  # noqa: E501
                plt.plot(sp.leff, y, color='b', ls="-", label=f"Filtered {mtyp}, B only")  # noqa: E501
                plt.fill_between(sp.leff, y-yerr, y+yerr, color='b', alpha=0.2)

                y = np.mean(np.array([cl[pols] for cl in cls_filtered_nopure]), axis=0)*cl2dl  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure]), axis=0)*cl2dl  # noqa: E501
                plt.plot(sp.leff, y, color='k', ls="-", label=f"Filtered {mtyp}, no purification")  # noqa: E501
                plt.fill_between(sp.leff, y-yerr, y+yerr, color='k', alpha=0.2)

                y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure]), axis=0)*cl2dl  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure]), axis=0)*cl2dl  # noqa: E501
                plt.plot(sp.leff, y, color='y', ls="-", label=f"Filtered {mtyp}, purified")  # noqa: E501
                plt.fill_between(sp.leff, y-yerr, y+yerr, color='y', alpha=0.2)

                y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_adv]), axis=0)*cl2dl  # noqa: E501
                yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_adv]), axis=0)*cl2dl  # noqa: E501
                plt.plot(sp.leff, y, color='g', ls="-", label=f"f{mtyp}, {purify_method}-purified {'(N='+str(nsims_purify)+')' if purify_method == 'deproj' else ''}")  # noqa: E501
                plt.fill_between(sp.leff, y-yerr, y+yerr, color='g', alpha=0.2)

                plt.fill_between(sp.leff, y-yerr, y+yerr, color='y', alpha=0.2)
                plt.plot(clth[pols][:sp.lmax+1]*cl2dl, 'r--', alpha=0.5, label="Theory")  # noqa: E501

                plt.xlim([2, lmax_plot])
                if pols == "BB":
                    plt.yscale('log')
                    ylim = (1e-6, 2e-3) if plot_dl else (1e-8, 1e-5)
                    # plt.ylim(ylim)
                elif pols == "EE":
                    plt.yscale('log')
                    # ylim = (1e-4, 1e0) if plot_dl else (1e-6, 1e-2)
                    plt.ylim(ylim)
                plt.xlabel(r"$\ell$", fontsize=14)
                lab = "D" if plot_dl else "C"
                plt.ylabel(fr"${{{lab}}}_\ell^{{{pols}}}$", fontsize=14)
                plt.legend()
                print(f"    PLOT SAVED {plot_dir}/cl_{mtyp}{pols}_filtered.pdf")  # noqa: E501
                plt.savefig(f"{plot_dir}/cl_{mtyp}{pols}_filtered.pdf", bbox_inches="tight")  # noqa: E501
                plt.close()
        print(f"     ALL PLOTS: {plot_dir}")


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
