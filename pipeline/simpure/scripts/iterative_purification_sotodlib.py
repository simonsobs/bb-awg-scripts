import numpy as np
import matplotlib.pyplot as plt
import os
import utils as ut
from simpure import SimPure

# Changelog:
# * 2025/12/19: added masked_on_input=True for deprojection spectra
# * 2025/12/20: made deprojection matrix with QU only
# * 2025/12/20: fixed bug in cls_filtered_nopure (purify was set to True)
# * 2026/01/17: fixed bug in get_pcls_mat_transfer (wrong field pair ordering)
# * 2026/01/25: ported functions to simpure
# * 2026/01/29: added flag tf_type to get_inv_coupling and compute_pspec
# * 2026/02/03: added flag overwrite_spectra


def main():
    print("  0. Initialization")
    # pixelization
    pix_type = "car"
    nside = 128
    base_dir = "/cephfs/soukdata/user_data/kwolz/simpure"
    filter_setup = "butter4_20251007"
    car_template = "/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/data/band_car_fejer1_20arcmin.fits"  # noqa: E501
    beam_fwhm = 30
    nlb = 10  # number of multipoles per bin

    # general
    nsims_purify = 800  # number of pure-E sims used for template deprojection
    nsims_cmb = 100  # number of validation sims
    nsims_transfer = 50  # number of pure (E,B sims used for transfer function
    id_sim_transfer_start = 0
    lmax_plot = 300
    overwrite = False  # If True, always recompute products.
    overwrite_spectra = True  # If True, update everything downstream of couplings step.
    deproject_null = False  # Deproject null vector instead of pureB template.
    ignore_filtering = False  # If True, only check mask-based purification.

    out_dir = f"/cephfs/soukdata/user_data/kwolz/simpure/purification/{filter_setup}_thresh20percent_20260203"  # noqa: E501
    plot_dir = f"{out_dir}/plots_ndep{nsims_purify}_20260203"
    mask_file = f"/cephfs/soukdata/user_data/kwolz/simpure/filtered_pure_sims/satp3/f090/{filter_setup}/mask_thresh20percent/masks/analysis_mask_apo10_C1_car.fits"  # noqa

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
    _, clth = ut.get_theory_cls(cosmo, lmax=sp.lmax+500, beam_fwhm=beam_fwhm)

    os.makedirs(plot_dir, exist_ok=True)
    ut.plot_map(sp.mask, file_name=f"{plot_dir}/mask", pix_type=pix_type)
    ut.plot_map(sp.mask_bin, file_name=f"{plot_dir}/mask_bin",
                pix_type=pix_type)

    if not ignore_filtering:
        print("  1. Make deprojection matrix")
        if not deproject_null:

            sim_dir = f"{out_dir}/sims_filt"
            sim_fn = "sim_pureE_alpha2_filt_pure_Bout_{id_sim:04d}.fits"
            mat_fn = f"{out_dir}/matcorr_alpha2_filt_pure_Bout_nsims{nsims_purify}.npz"  # noqa
            mat_plot_fn = f"{plot_dir}/M_deproject_eigvals_nsims{nsims_purify}.pdf"  # noqa
            mp_sims, mat = sp.make_deprojection_matrix(
                sim_dir, sim_fn, mat_fn, nsims_purify, overwrite, mat_plot_fn)
        else:
            print("  1A. Null B-residuals to mock deproject")
            mp = sp.load_cmb_sim(0, filtered=True)
            mp_sims = np.zeros((nsims_purify,) + mp.shape)
            mat = np.eye(nsims_purify)

        print("  2. Transfer function computation")
        print("  2A. TF sims unfiltered w/o purification")
        kwargs = {"filtered": False, "purified": False, "deprojected": False}
        cls_tf_unfiltered_nopure = sp.get_tf_sims(
            out_dir, nsims_transfer, overwrite=overwrite,
            id_sim_start=id_sim_transfer_start, **kwargs)

        print("  2B. TF sims unfiltered w/ purification")
        kwargs = {"filtered": False, "purified": True, "deprojected": False}
        cls_tf_unfiltered_pure = sp.get_tf_sims(
            out_dir, nsims_transfer, overwrite=overwrite,
            id_sim_start=id_sim_transfer_start, **kwargs)

        print("  2C. TF sims filtered w/o purification")
        kwargs = {"filtered": True, "purified": False, "deprojected": False}
        cls_tf_filtered_nopure = sp.get_tf_sims(
            out_dir, nsims_transfer, overwrite=overwrite,
            id_sim_start=id_sim_transfer_start, **kwargs)

        print("  2D. TF sims filtered w/ purification")
        kwargs = {"filtered": True, "purified": True, "deprojected": False}
        cls_tf_filtered_pure = sp.get_tf_sims(
            out_dir, nsims_transfer, overwrite=overwrite, 
            id_sim_start=id_sim_transfer_start, **kwargs)

        print("  2E. TF sims filtered w/ purification & deprojection")
        kwargs = {"filtered": True, "purified": True, "deprojected": True}
        cls_tf_filtered_pure_dep = sp.get_tf_sims(
            out_dir, nsims_transfer, overwrite=overwrite,
            mp_sims=mp_sims, mat=mat,
            id_sim_start=id_sim_transfer_start, **kwargs
        )

        print("  2F. TF w/ purification")
        transfer_pure = ut.get_transfer_dict(cls_tf_filtered_pure,
                                             cls_tf_unfiltered_pure)

        print("  2G. TF w/o purification")
        transfer_nopure = ut.get_transfer_dict(cls_tf_filtered_nopure,
                                               cls_tf_unfiltered_nopure)

        print("  2H. TF with purification & deprojection")
        transfer_pure_dep = ut.get_transfer_dict(cls_tf_filtered_pure_dep,
                                                 cls_tf_unfiltered_pure)
        np.save(f"{out_dir}/tf_pure_dep.npy", transfer_pure_dep["full_tf"])
        print(f"Saved TF: {out_dir}/tf_pure_dep.npy")

        field_pairs = ["EE", "EB", "BE", "BB"]
        file_name = f"{plot_dir}/transfer_dep_nsims{nsims_purify}_tf_nsims{nsims_transfer}.pdf"  # noqa
        tf_dict = {
            "nopure": transfer_nopure, "pure": transfer_pure,
            f"pure_dep (N={nsims_purify})": transfer_pure_dep
        }
        ut.plot_transfer_function(sp.leff, tf_dict, 0, lmax_plot, field_pairs,  # noqa
                                  file_name=file_name)
        ut.plot_transfer_function(sp.leff, tf_dict, 0, lmax_plot, ["BB"],
                                  file_name=file_name.replace(".pdf", "_BB.pdf"))  # noqa

    print("  3. Compute full couplings")
    _, bpw_msk_nopure = sp.get_inv_coupling(return_bp_win=True,
                                            overwrite=True)
    _, bpw_msk_pure = sp.get_inv_coupling(nmt_purify=True,
                                          return_bp_win=True,
                                          overwrite=True)
    if not ignore_filtering:
        _, bpw_fil_nopure = sp.get_inv_coupling(
            transfer=transfer_nopure["full_tf"],
            tf_type="nopure",
            nmt_purify=False,
            return_bp_win=True,
            overwrite=True
        )
        _, bpw_fil_pure = sp.get_inv_coupling(
            transfer=transfer_pure["full_tf"],
            tf_type="pure",
            nmt_purify=True,
            return_bp_win=True,
            overwrite=True
        )
        _, bpw_fil_pure_dep = sp.get_inv_coupling(
            transfer=transfer_pure_dep["full_tf"],
            tf_type="pure_dep",
            nmt_purify=True,
            return_bp_win=True,
            overwrite=True
        )

    clth_msk_nopure = ut.bin_theory_cls(clth, bpw_msk_nopure)
    clth_msk_pure = ut.bin_theory_cls(clth, bpw_msk_pure)
    if not ignore_filtering:
        clth_fil_nopure = ut.bin_theory_cls(clth, bpw_fil_nopure)
        clth_fil_pure = ut.bin_theory_cls(clth, bpw_fil_pure)
        clth_fil_pure_dep = ut.bin_theory_cls(clth, bpw_fil_pure_dep)

    # Compute CLs
    print("  4. Power spectrum computation")
    print("  4A. Masked cmbEB without purification")
    kwargs = {"noE": False, "filtered": False, "purified": False,
              "deprojected": False, "TFed": False}
    cls_masked_nopure = sp.get_cmb_spectra(
        out_dir, nsims_cmb, overwrite=overwrite, plot_dir=plot_dir,
        **kwargs)

    print("  4B. Masked cmbEB with purification")
    kwargs = {"noE": False, "filtered": False, "purified": True,
              "deprojected": False, "TFed": False}
    cls_masked_pure = sp.get_cmb_spectra(
        out_dir, nsims_cmb, overwrite=overwrite, plot_dir=plot_dir, **kwargs)

    print("  4C. Masked cmbB (without purification)")
    kwargs = {"noE": True, "filtered": False, "purified": False,
              "deprojected": False, "TFed": False}
    cls_noe_masked = sp.get_cmb_spectra(
        out_dir, nsims_cmb, overwrite=overwrite, plot_dir=plot_dir, **kwargs)

    if not ignore_filtering:
        print("  4D. Filtered cmbEB without purification")
        kwargs = {"noE": False, "filtered": True, "purified": False,
                  "deprojected": False, "TFed": False}
        cls_filtered_nopure = sp.get_cmb_spectra(
            out_dir, nsims_cmb, overwrite=overwrite, plot_dir=plot_dir,
            **kwargs
        )

        print("  4E. Filtered cmbEB without purification, TFed")
        kwargs = {"noE": False, "filtered": True, "purified": False,
                  "deprojected": False, "TFed": True}
        cls_filtered_nopure_tfed = sp.get_cmb_spectra(
            out_dir, nsims_cmb, overwrite=overwrite_spectra,
            tf=transfer_nopure["full_tf"], plot_dir=plot_dir, **kwargs
        )

        print("  4F. Filtered cmbEB with purification")
        kwargs = {"noE": False, "filtered": True, "purified": True,
                  "deprojected": False, "TFed": False}
        cls_filtered_pure = sp.get_cmb_spectra(
            out_dir, nsims_cmb, overwrite=overwrite, plot_dir=plot_dir,
            **kwargs
        )

        print("  4G. Fitered cmbEB with purification, TFed")
        kwargs = {"noE": False, "filtered": True, "purified": True,
                  "deprojected": False, "TFed": True}
        cls_filtered_pure_tfed = sp.get_cmb_spectra(
            out_dir, nsims_cmb, overwrite=overwrite_spectra,
            tf=transfer_pure["full_tf"], plot_dir=plot_dir, **kwargs
        )

        print("  4H. Filtered cmbB (without purification)")
        kwargs = {"noE": True, "filtered": True, "purified": False,
                  "deprojected": False, "TFed": False}
        cls_noe_filtered = sp.get_cmb_spectra(
            out_dir, nsims_cmb, overwrite=overwrite, plot_dir=plot_dir,
            **kwargs
        )

        print("  4J. Filtered cmbB (without purification), TFed")
        kwargs = {"noE": True, "filtered": True, "purified": False,
                  "deprojected": False, "TFed": True}
        cls_noe_filtered_tfed = sp.get_cmb_spectra(
            out_dir, nsims_cmb, overwrite=overwrite_spectra,
            tf=transfer_nopure["full_tf"], plot_dir=plot_dir, **kwargs
        )

        print("  4K. Filtered with purification and deprojection")
        kwargs = {"noE": False, "filtered": True, "purified": True,
                  "deprojected": True, "TFed": False}
        cls_filtered_pure_dep = sp.get_cmb_spectra(
            out_dir, nsims_cmb, overwrite=overwrite, mp_sims=mp_sims, mat=mat,
            plot_dir=plot_dir, **kwargs
        )

        print("  4L. Filtered with purification and deprojection, TFed")
        kwargs = {"noE": False, "filtered": True, "purified": True,
                  "deprojected": True, "TFed": True}
        cls_filtered_pure_dep_tfed = sp.get_cmb_spectra(
            out_dir, nsims_cmb, overwrite=overwrite_spectra,
            tf=transfer_pure_dep["full_tf"], mp_sims=mp_sims, mat=mat,
            plot_dir=plot_dir, **kwargs
        )

    # Plotting
    print("  5. Plotting")
    os.makedirs(f"{out_dir}/plots", exist_ok=True)
    plot_dl = False

    plt.figure()
    plt.title('Cl, no filtering')

    cl2dl = sp.leff*(sp.leff+1)/2./np.pi if plot_dl else 1.
    y = np.mean(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)*cl2dl
    yerr = np.std(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)*cl2dl  # noqa
    plt.plot(sp.leff, y, color='b', ls="-", label="Masked CMB, B only")
    plt.fill_between(sp.leff, y-yerr, y+yerr, color='b', alpha=0.2)

    y = np.mean(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)*cl2dl  # noqa
    yerr = np.std(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)*cl2dl  # noqa: E501
    plt.plot(sp.leff, y, color='k', ls="-", label="Masked CMB, no purification")  # noqa: E501
    plt.fill_between(sp.leff, y-yerr, y+yerr, color='k', alpha=0.2)

    y = np.mean(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)*cl2dl
    yerr = np.std(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)*cl2dl
    plt.plot(sp.leff, y, color='y', ls="-", label="Masked CMB, purified")
    plt.fill_between(sp.leff, y-yerr, y+yerr, color='y', alpha=0.2)
    ls = np.arange(sp.lmax+1)
    clth2dl = ls*(ls+1)/2./np.pi if plot_dl else 1.
    plt.plot(clth["BB"][:sp.lmax+1]*clth2dl, 'r--', alpha=0.5, label="Theory")

    plt.xlim([2, lmax_plot])
    ylim = (1e-6, 2e-3) if plot_dl else (1e-8, 1e-5)
    plt.ylim(ylim)
    plt.xlabel(r"$\ell$", fontsize=14)
    ylab = r"$D_\ell^{BB}$" if plot_dl else r"$C_\ell^{BB}$"
    plt.ylabel(ylab, fontsize=14)
    plt.yscale('log')
    plt.legend()
    print(f"  PLOT SAVED {plot_dir}/cl_masked_BB.pdf")
    plt.savefig(f"{plot_dir}/cl_masked_BB.pdf", bbox_inches="tight")
    plt.close()

    thbb = sp.bins.bin_cell(clth["BB"][:sp.bins.lmax+1])
    y = np.mean(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)*cl2dl
    yerr = np.std(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)*cl2dl  # noqa
    plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)), 'b-', alpha=0.5, label="Masked CMB, B only")  # noqa: E501

    thbb = clth_msk_nopure["BB"]
    y = np.mean(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)*cl2dl  # noqa
    yerr = np.std(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)*cl2dl  # noqa: E501
    plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)), 'k-', alpha=0.5, label="Masked CMB, no purification")  # noqa: E501

    thbb = clth_msk_pure["BB"]
    y = np.mean(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)*cl2dl
    yerr = np.std(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)*cl2dl
    plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)), 'y-', alpha=0.5, label="Masked CMB, purified")  # noqa: E501
    plt.axhline(0, color="r", ls="--")

    plt.xlim([2, lmax_plot])
    plt.ylim((-20, 20))
    plt.xlabel(r"$\ell$", fontsize=14)
    plt.ylabel(r"$(\hat{C}_\ell^{BB} - C_\ell^{BB,\, th})/(\sigma(C_\ell^{BB})/\sqrt{N_{\rm sims}})$", fontsize=14)  # noqa: E501
    plt.legend()
    print(f"    PLOT SAVED {plot_dir}/bias_masked_BB.pdf")
    plt.savefig(f"{plot_dir}/bias_masked_BB.pdf", bbox_inches="tight")
    plt.close()

    yerr_ref = np.std(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)
    yerr = np.std(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)
    plt.plot(sp.leff, yerr/yerr_ref, 'b-', alpha=0.5, label="Masked CMB, purified")  # noqa: E501
    sigma_masked = yerr/yerr_ref

    yerr = np.std(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)
    plt.plot(sp.leff, yerr/yerr_ref, 'k-', alpha=0.5, label="Masked CMB, no purification")  # noqa: E501
    plt.axhline(1, color="k", ls="--")

    plt.xlim([2, lmax_plot])
    plt.ylim((0.5, 1000))
    plt.xlabel(r"$\ell$", fontsize=14)
    plt.ylabel(r"$\sigma(C_\ell^{BB, X})/\sigma(C_\ell^{BB,\,\rm{B only}})$", fontsize=14)  # noqa: E501
    plt.yscale('log')
    plt.legend()
    print(f"    PLOT SAVED {plot_dir}/error_masked_BB.pdf")
    plt.savefig(f"{plot_dir}/error_masked_BB.pdf", bbox_inches="tight")
    plt.close()

    if not ignore_filtering:
        for pols in ["EE", "EB", "BB"]:
            plt.title('Cl, w/ filtering, TFed')

            y = np.mean(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)*cl2dl  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)*cl2dl  # noqa: E501
            plt.plot(sp.leff, y, color='b', ls="-", label="Filtered CMB, B only")  # noqa: E501
            plt.fill_between(sp.leff, y-yerr, y+yerr, color='b', alpha=0.2)

            y = np.mean(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)*cl2dl  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)*cl2dl  # noqa: E501
            plt.plot(sp.leff, y, color='k', ls="-", label="Filtered CMB, no purification")  # noqa: E501
            plt.fill_between(sp.leff, y-yerr, y+yerr, color='k', alpha=0.2)

            y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)*cl2dl  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)*cl2dl  # noqa: E501
            plt.plot(sp.leff, y, color='y', ls="-", label="Filtered CMB, purified")  # noqa: E501
            plt.fill_between(sp.leff, y-yerr, y+yerr, color='y', alpha=0.2)

            y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_dep_tfed]), axis=0)*cl2dl  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_dep_tfed]), axis=0)*cl2dl  # noqa: E501
            plt.plot(sp.leff, y, color='g', ls="-", label="fCMB, purified & deproj.")  # noqa: E501
            plt.fill_between(sp.leff, y-yerr, y+yerr, color='g', alpha=0.2)

            plt.fill_between(sp.leff, y-yerr, y+yerr, color='y', alpha=0.2)
            plt.plot(clth[pols][:sp.lmax+1]*clth2dl, 'r--', alpha=0.5, label="Theory")  # noqa: E501
            plt.plot(sp.leff, clth_fil_pure_dep[pols], 'r-.', alpha=0.5, label="Theory dep.")  # noqa: E501

            plt.xlim([2, lmax_plot])
            plt.xlabel(r"$\ell$", fontsize=14)
            ylab = fr"$D_\ell^{{{pols}}}$" if plot_dl else fr"$C_\ell^{{{pols}}}$"  # noqa: E501
            plt.ylabel(ylab, fontsize=14)
            if pols == "BB":
                plt.yscale('log')
                ylim = (1e-6, 2e-3) if plot_dl else (1e-8, 1e-5)
                plt.ylim(ylim)
            elif pols == "EE":
                plt.yscale('log')
                ylim = (1e-4, 1e0) if plot_dl else (1e-6, 1e-2)
                plt.ylim(ylim)
            plt.legend()
            print(f"    PLOT SAVED {plot_dir}/cl_filtered_tfed_{pols}.pdf")
            plt.savefig(f"{plot_dir}/cl_filtered_tfed_{pols}.pdf", bbox_inches="tight")  # noqa: E501
            plt.close()

        for pols in ["EE", "EB", "BB"]:
            thbb = sp.bins.bin_cell(clth[pols][:sp.bins.lmax+1])
            y = np.mean(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)  # noqa: E501
            plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)), 'b-', alpha=0.5, label="Filtered CMB, B only")  # noqa: E501

            thbb = clth_fil_nopure[pols]
            y = np.mean(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)  # noqa: E501
            plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)), 'k-', alpha=0.5, label="Filtered CMB, no purification")  # noqa: E501

            thbb = clth_fil_pure[pols]
            y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)  # noqa: E501
            plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)), 'y-', alpha=0.5, label="Filtered CMB, purified")  # noqa: E501

            thbb = clth_fil_pure_dep[pols]
            y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_dep_tfed]), axis=0)  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_dep_tfed]), axis=0)  # noqa: E501
            plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)), 'g-', alpha=0.5, label=f"fCMB, purif. & deproj. (N={nsims_purify})")  # noqa: E501
            plt.axhline(0, color="r", ls="--")

            plt.xlim([2, lmax_plot])
            plt.ylim((-20, 20))
            plt.xlabel(r"$\ell$", fontsize=14)
            plt.ylabel(fr"$(\hat{{C}}_\ell^{{{pols}}} - C_\ell^{{{pols},\, th}})/(\sigma(C_\ell^{{{pols}}})/\sqrt{{N_{{\rm sims}}}})$", fontsize=14)  # noqa: E501
            plt.legend()
            print(f"    PLOT SAVED {plot_dir}/bias_filtered_tfed_{pols}.pdf")  # noqa: E501
            plt.savefig(f"{plot_dir}/bias_filtered_tfed_{pols}.pdf", bbox_inches="tight")  # noqa: E501
            plt.close()

        for pols in ["EE", "EB", "BB"]:
            yerr_ref = np.std(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)  # noqa: E501
            plt.plot(sp.leff, sigma_masked, 'm-', alpha=0.5, label="Masked CMB, purified")  # noqa: E501
            plt.plot(sp.leff, yerr/yerr_ref, 'k-', alpha=0.5, label="Filtered CMB, no purification")  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)  # noqa: E501
            plt.plot(sp.leff, yerr/yerr_ref, 'y-', alpha=0.5, label="Filtered CMB, purified")  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_dep_tfed]), axis=0)  # noqa: E501
            plt.plot(sp.leff, yerr/yerr_ref, 'g-', alpha=0.5, label="fCMB, purified & deproj.")  # noqa: E501
            plt.axhline(1, color="b", linestyle="--", alpha=0.5, label="Filtered CMB, B only")  # noqa: E501

            plt.xlim([2, lmax_plot])
            plt.ylim((0.5, 1000))
            plt.xlabel(r"$\ell$", fontsize=14)
            plt.ylabel(fr"$\sigma(C_\ell^{{{pols}, X}})/\sigma(C_\ell^{{{pols}}},\,\rm{{B only}})$", fontsize=14)  # noqa: E501
            plt.yscale('log')
            plt.legend()
            print(f"    PLOT SAVED {plot_dir}/error_filtered_tfed_{pols}.pdf")  # noqa: E501
            plt.savefig(f"{plot_dir}/error_filtered_tfed_{pols}.pdf", bbox_inches="tight")  # noqa: E501
            plt.close()

        for pols in ["EE", "EB", "BB"]:
            yerr = np.std(np.array([cl[pols] for cl in cls_noe_masked]), axis=0)*cl2dl  # noqa: E501
            plt.plot(sp.leff, yerr, 'm--', alpha=0.5, label="Masked CMB, B only")  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_noe_filtered_tfed]), axis=0)*cl2dl  # noqa: E501
            plt.plot(sp.leff, yerr, 'b--', alpha=0.5, label="Filtered CMB, B only")  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure_tfed]), axis=0)*cl2dl  # noqa: E501
            plt.plot(sp.leff, yerr, 'k-', alpha=0.5, label="Filtered CMB, no purification")  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_tfed]), axis=0)*cl2dl  # noqa: E501
            plt.plot(sp.leff, yerr, 'y-', alpha=0.5, label="Filtered CMB, purified")  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_dep_tfed]), axis=0)*cl2dl  # noqa: E501
            plt.plot(sp.leff, yerr, 'g-', alpha=0.5, label="fCMB, purified & deproj.")  # noqa: E501

            plt.xlim([2, lmax_plot])
            plt.yscale("log")
            plt.xlabel(r"$\ell$", fontsize=14)
            lab = "D" if plot_dl else "C"
            plt.ylabel(ylab, fontsize=14)
            plt.ylabel(fr"$\sigma({{{lab}}}_\ell^{{{pols}, X}})$", fontsize=14)
            plt.yscale('log')
            plt.legend()
            print(f"    PLOT SAVED {plot_dir}/abserr_filtered_tfed_{pols}.pdf")  # noqa: E501
            plt.savefig(f"{plot_dir}/abserr_filtered_tfed_{pols}.pdf", bbox_inches="tight")  # noqa: E501
            plt.close()

        plt.title('Cl, w/ filtering not TFed')
        for pols in ["EE", "EB", "BB"]:
            y = np.mean(np.array([cl[pols] for cl in cls_noe_filtered]), axis=0)*cl2dl  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_noe_filtered]), axis=0)*cl2dl  # noqa: E501
            plt.plot(sp.leff, y, color='b', ls="-", label="Filtered CMB, B only")  # noqa: E501
            plt.fill_between(sp.leff, y-yerr, y+yerr, color='b', alpha=0.2)

            y = np.mean(np.array([cl[pols] for cl in cls_filtered_nopure]), axis=0)*cl2dl  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure]), axis=0)*cl2dl  # noqa: E501
            plt.plot(sp.leff, y, color='k', ls="-", label="Filtered CMB, no purification")  # noqa: E501
            plt.fill_between(sp.leff, y-yerr, y+yerr, color='k', alpha=0.2)

            y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure]), axis=0)*cl2dl  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure]), axis=0)*cl2dl  # noqa: E501
            plt.plot(sp.leff, y, color='y', ls="-", label="Filtered CMB, purified")  # noqa: E501
            plt.fill_between(sp.leff, y-yerr, y+yerr, color='y', alpha=0.2)

            y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_dep]), axis=0)*cl2dl  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_dep]), axis=0)*cl2dl  # noqa: E501
            plt.plot(sp.leff, y, color='g', ls="-", label=f"fCMB, purified & deproj. (N={nsims_purify})")  # noqa: E501
            plt.fill_between(sp.leff, y-yerr, y+yerr, color='g', alpha=0.2)

            plt.fill_between(sp.leff, y-yerr, y+yerr, color='y', alpha=0.2)
            plt.plot(clth[pols][:sp.lmax+1]*clth2dl, 'r--', alpha=0.5, label="Theory")  # noqa: E501

            plt.xlim([2, lmax_plot])
            if pols == "BB":
                plt.yscale('log')
                ylim = (1e-6, 2e-3) if plot_dl else (1e-8, 1e-5)
                plt.ylim(ylim)
            elif pols == "EE":
                plt.yscale('log')
                ylim = (1e-4, 1e0) if plot_dl else (1e-6, 1e-2)
                plt.ylim(ylim)
            plt.xlabel(r"$\ell$", fontsize=14)
            lab = "D" if plot_dl else "C"
            plt.ylabel(fr"${{{lab}}}_\ell^{{{pols}}}$", fontsize=14)
            plt.legend()
            print(f"    PLOT SAVED {plot_dir}/cl_{pols}_filtered.pdf")
            plt.savefig(f"{plot_dir}/cl_{pols}_filtered.pdf", bbox_inches="tight")  # noqa: E501
            plt.close()

        for pols in ["EE", "EB", "BB"]:
            thbb = sp.bins.bin_cell(clth[pols][:sp.bins.lmax+1])
            y = np.mean(np.array([cl[pols] for cl in cls_noe_filtered]), axis=0)  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_noe_filtered]), axis=0)  # noqa: E501
            plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)), 'b-', alpha=0.5, label="Filtered CMB, B only")  # noqa: E501

            thbb = clth_fil_nopure[pols]
            y = np.mean(np.array([cl[pols] for cl in cls_filtered_nopure]), axis=0)  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure]), axis=0)  # noqa: E501
            plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)), 'k-', alpha=0.5, label="Filtered CMB, no purification")  # noqa: E501

            thbb = clth_fil_pure[pols]
            y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure]), axis=0)  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure]), axis=0)  # noqa: E501
            plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)), 'y-', alpha=0.5, label="Filtered CMB, purified")  # noqa: E501

            thbb = clth_fil_pure_dep[pols]
            y = np.mean(np.array([cl[pols] for cl in cls_filtered_pure_dep]), axis=0)  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_dep]), axis=0)  # noqa: E501
            plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)), 'g-', alpha=0.5, label=f"fCMB, purified & deproj. (N={nsims_purify})")  # noqa: E501
            plt.axhline(0, color="r", ls="--")

            plt.xlim([2, lmax_plot])
            plt.ylim((-20, 20))
            plt.xlabel(r"$\ell$", fontsize=14)
            plt.ylabel(fr"$(\hat{{C}}_\ell^{{{pols}}} - C_\ell^{{{pols},\, th}})/(\sigma(C_\ell^{{{pols}}})/\sqrt{{N_{{\rm sims}}}})$",  # noqa: E501
                       fontsize=14)
            plt.legend()
            print(f"    PLOT SAVED {plot_dir}/bias_filtered_{pols}.pdf")
            plt.savefig(f"{plot_dir}/bias_filtered_{pols}.pdf", bbox_inches="tight")  # noqa: E501
            plt.close()

        for pols in ["EE", "EB", "BB"]:
            yerr_ref = np.std(np.array([cl[pols] for cl in cls_noe_filtered]), axis=0)  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_nopure]), axis=0)  # noqa: E501
            plt.plot(sp.leff, sigma_masked, 'm-', alpha=0.5, label="Masked CMB, purified")  # noqa: E501
            plt.plot(sp.leff, yerr/yerr_ref, 'k-', alpha=0.5, label="Filtered CMB, no purification")  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure]), axis=0)  # noqa: E501
            plt.plot(sp.leff, yerr/yerr_ref, 'y-', alpha=0.5, label="Filtered CMB, purified")  # noqa: E501
            yerr = np.std(np.array([cl[pols] for cl in cls_filtered_pure_dep]), axis=0)  # noqa: E501
            plt.plot(sp.leff, yerr/yerr_ref, 'g-', alpha=0.5, label=f"fCMB, purified & deproj. (N={nsims_purify})")  # noqa: E501
            plt.axhline(1, color="b", ls="--")

            plt.xlim([2, lmax_plot])
            plt.ylim((0.5, 1000))
            plt.xlabel(r"$\ell$", fontsize=14)
            plt.ylabel(fr"$\sigma(C_\ell^{{{pols}, X}})/\sigma(C_\ell^{{{pols}}},\,\rm{{B only}})$", fontsize=14)  # noqa: E501
            plt.yscale('log')
            plt.legend()
            print(f"    PLOT SAVED {plot_dir}/error_filtered_{pols}.pdf")
            plt.savefig(f"{plot_dir}/error_filtered_{pols}.pdf", bbox_inches="tight")  # noqa: E501
            plt.close()
        print(f"     ALL PLOTS: {plot_dir}")


if __name__ == "__main__":
    main()
