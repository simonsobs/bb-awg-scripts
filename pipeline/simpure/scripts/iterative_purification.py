import numpy as np
import matplotlib.pyplot as plt
import os
import utils as ut
from simpure import SimPure

# Changelog:
# * 2025/12/19: added masked_on_input=True for deprojection spectra


def main():
    print("  0. Initialization")
    # pixelization
    pix_type = "hp"
    nside = 128
    base_dir = "/pscratch/sd/k/kwolz/bbdev/simpure"
    filter_setup = "obsmat_apo_nside128"
    car_template = "/shared_home/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/data/band_car_fejer1_20arcmin.fits"  # noqa: E501
    beam_fwhm = 30

    # general
    nsims_purify = 1000  # number of pure-E sims used for template deprojection
    nsims_cmb = 100  # number of validation sims
    nsims_transfer = 100  # number of pure (E,B sims used for transfer function
    id_sim_transfer_start = 0
    lmax_plot = 300
    overwrite = True  # If True, always recompute products.
    deproject_null = False  # Deproject null vector instead of pureB template.
    ignore_filtering = False  # If True, only check mask-based purification.

    out_dir = f"/pscratch/sd/k/kwolz/bbdev/simpure/purification/{filter_setup}_20251219"  # noqa: E501
    plot_dir = f"{out_dir}/plots_ndep{nsims_purify}_ntrf{nsims_transfer}_20251219"  # noqa: E501
    mask_file = "/global/homes/k/kwolz/bbdev/bb-awg-scripts/pipeline/simpure/data/mask_apo_nside128.fits"  # noqa: E501

    sp = SimPure(pix_type,
                 base_dir,
                 out_dir,
                 filter_setup,
                 mask_file,
                 nside=nside,
                 car_template=car_template,
                 beam_fwhm=beam_fwhm,
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
            # Map B-residuals (B-only component of filtered,
            # mask-purified pure-E simulations)
            print("  1A. Mapping B-residuals from filtered pure-E sims")

            os.makedirs(f"{out_dir}/sims_filt", exist_ok=True)
            mp_sims = []

            for i in range(nsims_purify):
                if i % 50 == 0:
                    print("   ", i)
                fname = f"{out_dir}/sims_filt/sim_pureE_alpha2_filt_pure_Bout_{i:04d}.fits"  # noqa: E501
                if os.path.isfile(fname) and not overwrite:
                    mp_masked_bonly = ut.read_map(
                        fname,
                        pix_type=pix_type,
                        fields_hp=[0, 1, 2],
                        car_template=car_template
                    )
                else:
                    # Load filtered pure-E sim
                    mp = sp.load_purification_sim(i, filtered=True)
                    if i == 0:
                        ut.plot_map(
                            mp,
                            file_name=f"{plot_dir}/sim_pureE_alpha2_filt_{i:04d}",  # noqa: E501
                            pix_type=pix_type
                        )
                    # Mask-purify it
                    mp_masked = sp.get_masked_map(mp, nmt_purify=True,
                                                  binary=False)
                    if i == 0:
                        ut.plot_map(
                            mp_masked,
                            file_name=f"{plot_dir}/sim_pureE_alpha2_filt_msk_purify_{i:04d}",  # noqa: E501
                            pix_type=pix_type
                        )
                    # Extract B-modes and remap them
                    mp_masked_bonly = sp.extract_pure_mode(mp_masked,
                                                           3*nside, "B")
                    if i == 0:
                        ut.plot_map(
                            mp_masked_bonly,
                            file_name=f"{plot_dir}/sim_pureE_alpha2_filt_pure_Bout_{i:04d}",  # noqa: E501
                            pix_type=pix_type
                        )
                    ut.write_map(fname, mp_masked_bonly, pix_type=pix_type)

                mp_sims.append(mp_masked_bonly)
            mp_sims = np.array(mp_sims)

            # Save M_ij = s_ipn *s_jpn, where s is the simulation vector
            # of B-residuals
            print("  1B. Making deprojection matrix from mapped B-residual sims")  # noqa: E501

            fname = f"{out_dir}/matcorr_alpha2_filt_pure_Bout_nsims{nsims_purify}.npz"  # noqa: E501
            if os.path.isfile(fname) and not overwrite:
                mat = np.load(fname, allow_pickle=True)['mat']
            else:
                mat = []
                for i, s in enumerate(mp_sims):
                    if i % 10 == 0:
                        print("   ", i)
                    if pix_type == "hp":
                        mat.append(np.sum(mp_sims*s[None, :, :], axis=(1, 2)))
                    else:
                        mat.append(np.sum(mp_sims*s[None, :, :, :],
                                          axis=(1, 2, 3)))
                mat = np.array(mat)
                np.savez(fname, mat=mat)

            # Visualize eigenvalues of M
            w, _ = np.linalg.eigh(mat)
            plt.plot(w[::-1])
            plt.yscale('log')
            plot_file = f"{plot_dir}/M_deproject_eigvals_nsims{nsims_purify}.pdf"  # noqa: E501
            plt.savefig(plot_file, bbox_inches="tight")
            print(f"    PLOT SAVED {plot_file}")
            plt.close()
        else:
            print("  1A. Null B-residuals to mock deproject")
            mp = sp.load_cmb_sim(0, filtered=True)
            mp_sims = np.zeros((nsims_purify,) + mp.shape)
            mat = np.eye(nsims_purify)

        print("  2. Transfer function computation")
        print("  2A. TF sims unfiltered w/o purification")
        fname = f"{out_dir}/cls_tf_unfiltered_nopure_nsims{nsims_transfer}.npz"
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
                    mp = sp.load_transfer_sim(i, filtered=False, typ=typ)
                    f = sp.compute_pspec(
                        mp, nmt_purify=False, return_just_field=True
                    )
                    fields[typ] = f
                    fields2[typ] = f
                pcls_mat = sp.get_pcls_mat_transfer(fields, fields2)
                cls_tf_unfiltered_nopure.append(pcls_mat)
            cls_tf_unfiltered_nopure = np.array(cls_tf_unfiltered_nopure)
            np.savez(fname, cls=cls_tf_unfiltered_nopure)

        print("  2B. TF sims unfiltered w/ purification")
        fname = out_dir + f"/cls_tf_unfiltered_pure_nsims{nsims_transfer}.npz"
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
                    mp = sp.load_transfer_sim(i, filtered=False, typ=typ)
                    f = sp.compute_pspec(
                        mp, nmt_purify=True, return_just_field=True
                    )
                    fields[typ] = f
                    fields2[typ] = f
                pcls_mat = sp.get_pcls_mat_transfer(fields, fields2)
                cls_tf_unfiltered_pure.append(pcls_mat)
            cls_tf_unfiltered_pure = np.array(cls_tf_unfiltered_pure)
            np.savez(fname, cls=cls_tf_unfiltered_pure)

        print("  2C. TF sims filtered w/o purification")
        fname = out_dir + f"/cls_tf_filtered_nopure_nsims{nsims_transfer}.npz"
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
                    mp = sp.load_transfer_sim(i, filtered=True, typ=typ)
                    f = sp.compute_pspec(
                        mp, nmt_purify=False, return_just_field=True
                    )
                    fields[typ] = f
                    fields2[typ] = f
                pcls_mat = sp.get_pcls_mat_transfer(fields, fields2)
                cls_tf_filtered_nopure.append(pcls_mat)
            cls_tf_filtered_nopure = np.array(cls_tf_filtered_nopure)
            np.savez(fname, cls=cls_tf_filtered_nopure)

        print("  2D. TF sims filtered w/ purification")
        fname = out_dir + f"/cls_tf_filtered_pure_nsims{nsims_transfer}.npz"
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
                    mp = sp.load_transfer_sim(i, filtered=True, typ=typ)
                    f = sp.compute_pspec(
                        mp, nmt_purify=True, return_just_field=True
                    )
                    fields[typ] = f
                    fields2[typ] = f
                pcls_mat = sp.get_pcls_mat_transfer(fields, fields2)
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
                    mp = sp.load_transfer_sim(i, filtered=True, typ=typ)
                    mp_masked = sp.get_masked_map(mp, nmt_purify=True,
                                                  binary=False)
                    mp_masked_dep, _ = ut.deproject_many(mp_masked,
                                                         mp_sims,
                                                         mat)
                    f = sp.compute_pspec(
                        mp_masked_dep,
                        return_just_field=True,
                        masked_on_input=True
                    )
                    fields[typ] = f
                    fields2[typ] = f
                pcls_mat = sp.get_pcls_mat_transfer(fields, fields2)
                cls_tf_filtered_pure_dep.append(pcls_mat)
            cls_tf_filtered_pure_dep = np.array(cls_tf_filtered_pure_dep)
            np.savez(fname, cls=cls_tf_filtered_pure_dep)

        print("  2F. TF w/ purification")
        fp_dum = ("None", "None")
        pcls_mat_filt_dict = {fp_dum: {"filtered": cls_tf_filtered_pure}}
        mean_pcls_mat_filt_dict = {fp_dum:
                                   np.mean(cls_tf_filtered_pure, axis=0)}
        mean_pcls_mat_unfilt_dict = {fp_dum:
                                     np.mean(cls_tf_unfiltered_pure, axis=0)}
        transfer_pure = ut.get_transfer_dict(mean_pcls_mat_filt_dict,
                                             mean_pcls_mat_unfilt_dict,
                                             pcls_mat_filt_dict,
                                             [fp_dum])[fp_dum]

        print("  2G. TF w/o purification")
        pcls_mat_filt_dict = {fp_dum: {"filtered": cls_tf_filtered_nopure}}
        mean_pcls_mat_filt_dict = {fp_dum:
                                   np.mean(cls_tf_filtered_nopure, axis=0)}
        mean_pcls_mat_unfilt_dict = {fp_dum:
                                     np.mean(cls_tf_unfiltered_nopure, axis=0)}
        transfer_nopure = ut.get_transfer_dict(mean_pcls_mat_filt_dict,
                                               mean_pcls_mat_unfilt_dict,
                                               pcls_mat_filt_dict,
                                               [fp_dum])[fp_dum]

        print("  2H. TF with purification & deprojection")
        pcls_mat_filt_dict = {fp_dum: {"filtered": cls_tf_filtered_pure_dep}}
        mean_pcls_mat_filt_dict = {fp_dum:
                                   np.mean(cls_tf_filtered_pure_dep, axis=0)}
        mean_pcls_mat_unfilt_dict = {fp_dum:
                                     np.mean(cls_tf_unfiltered_pure, axis=0)}
        transfer_pure_dep = ut.get_transfer_dict(mean_pcls_mat_filt_dict,
                                                 mean_pcls_mat_unfilt_dict,
                                                 pcls_mat_filt_dict,
                                                 [fp_dum])[fp_dum]

        field_pairs = ["EE", "EB", "BE", "BB"]
        file_name = f"{plot_dir}/transfer_dep_nsims{nsims_purify}_tf_nsims{nsims_transfer}.pdf"  # noqa
        tf_dict = {
            "nopure": transfer_nopure, "pure": transfer_pure,
            f"pure_dep (N={nsims_purify})": transfer_pure_dep
        }

        ut.plot_transfer_function(sp.leff, tf_dict, 0, lmax_plot, field_pairs,
                                  file_name=file_name)

    print("  3. Compute full couplings")
    _, bpw_msk_nopure = sp.get_inv_coupling(return_bp_win=True,
                                            overwrite=overwrite)
    _, bpw_msk_pure = sp.get_inv_coupling(nmt_purify=True,
                                          return_bp_win=True,
                                          overwrite=overwrite)
    if not ignore_filtering:
        _, bpw_fil_nopure = sp.get_inv_coupling(
            transfer=transfer_nopure["full_tf"],
            nmt_purify=False,
            return_bp_win=True,
            overwrite=overwrite
        )
        _, bpw_fil_pure = sp.get_inv_coupling(
            transfer=transfer_pure["full_tf"],
            nmt_purify=True,
            return_bp_win=True,
            overwrite=overwrite
        )

    clth_msk_nopure = ut.bin_theory_cls(clth, bpw_msk_nopure)
    clth_msk_pure = ut.bin_theory_cls(clth, bpw_msk_pure)
    if not ignore_filtering:
        clth_fil_nopure = ut.bin_theory_cls(clth, bpw_fil_nopure)
        clth_fil_pure = ut.bin_theory_cls(clth, bpw_fil_pure)

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
            mp = sp.load_cmb_sim(i, filtered=False)
            cl = sp.compute_pspec(mp)
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
            mp = sp.load_cmb_sim(i, filtered=False)
            if i == 0:
                ut.plot_map(
                    mp,
                    file_name=f"{plot_dir}/sim_cmb_{i:04d}",
                    pix_type=pix_type
                )
                ut.plot_map(
                    sp.get_masked_map(mp, nmt_purify=False, binary=False),
                    file_name=f"{plot_dir}/sim_cmb_masked_{i:04d}",
                    pix_type=pix_type
                )
                ut.plot_map(
                    sp.get_masked_map(mp, nmt_purify=True, binary=False),
                    file_name=f"{plot_dir}/sim_cmb_masked_purif_{i:04d}",
                    pix_type=pix_type
                )
            cl = sp.compute_pspec(mp, nmt_purify=True)
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
            mp = sp.load_cmb_sim(i, pols_keep="B")
            cl = sp.compute_pspec(mp)
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
                mp = sp.load_cmb_sim(i, filtered=True)
                if i == 0:
                    ut.plot_map(
                        mp,
                        file_name=f"{plot_dir}/sim_cmb_filt_{i:04d}",
                        pix_type=pix_type
                    )
                    ut.plot_map(
                        sp.get_masked_map(mp, nmt_purify=False, binary=False),
                        file_name=f"{plot_dir}/sim_cmb_filt_masked_{i:04d}",
                        pix_type=pix_type
                    )
                cl = sp.compute_pspec(mp, transfer=None, nmt_purify=True)
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
                mp = sp.load_cmb_sim(i, filtered=True)
                if i == 0:
                    ut.plot_map(
                        mp,
                        file_name=f"{plot_dir}/sim_cmb_filt_{i:04d}",
                        pix_type=pix_type
                    )
                    ut.plot_map(
                        sp.get_masked_map(mp, nmt_purify=False, binary=False),
                        file_name=f"{plot_dir}/sim_cmb_filt_masked_{i:04d}",
                        pix_type=pix_type
                    )
                cl = sp.compute_pspec(
                    mp,
                    transfer=transfer_nopure["full_tf"],
                    nmt_purify=False
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
                mp = sp.load_cmb_sim(i, filtered=True)
                cl = sp.compute_pspec(mp, transfer=None, nmt_purify=True)
                if i == 0:
                    ut.plot_map(
                        mp,
                        file_name=f"{plot_dir}/sim_cmb_filt_purify_{i:04d}",
                        pix_type=sp.pix_type
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
                mp = sp.load_cmb_sim(i, filtered=True)
                cl = sp.compute_pspec(
                    mp,
                    transfer=transfer_pure["full_tf"],
                    nmt_purify=True
                )

                if i == 0:
                    ut.plot_map(
                        mp, file_name=f"{plot_dir}/sim_cmb_filt_purify_{i:04d}",  # noqa
                        pix_type=sp.pix_type
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
                mp = sp.load_cmb_sim(i, filtered=True, pols_keep="B")
                cl = sp.compute_pspec(mp, transfer=None)
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
                mp = sp.load_cmb_sim(i, filtered=True, pols_keep="B")
                cl = sp.compute_pspec(mp, transfer=transfer_nopure["full_tf"])
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
                mp = sp.load_cmb_sim(i, filtered=True)
                mp_masked = sp.get_masked_map(mp, nmt_purify=True,
                                              binary=False)
                mp_masked_dep, mp_template = ut.deproject_many(mp_masked,
                                                               mp_sims,
                                                               mat)
                if i == 1:
                    ut.plot_map(
                        mp_template,
                        file_name=f"{plot_dir}/sim_cmb_filt_template_{i:04d}",
                        pix_type=pix_type
                    )
                    ut.plot_map(
                        mp_masked,
                        file_name=f"{plot_dir}/sim_cmb_filt_purify_{i:04d}",
                        pix_type=pix_type
                    )
                    ut.plot_map(
                        mp_masked_dep,
                        file_name=f"{plot_dir}/sim_cmb_filt_purify_dep_{i:04d}",  # noqa
                        pix_type=pix_type
                    )
                cl = sp.compute_pspec(
                    mp_masked_dep,
                    transfer=None,
                    nmt_purify=False,
                    nmt_purify_mcm=True,
                    masked_on_input=True
                )
                cls_filtered_pure_dep.append(cl)

            cls_filtered_pure_dep = np.array(cls_filtered_pure_dep)
            np.savez(fname, cls=cls_filtered_pure_dep)

        print("  4L. Filtered with purification and deprojection, TFed")
        fname = f"{out_dir}/cls_filtered_pure_dep_tfed_nsims{nsims_purify}.npz"
        if os.path.isfile(fname) and not overwrite:
            cls_filtered_pure_dep_tfed = np.load(fname,
                                                 allow_pickle=True)['cls']
        else:
            cls_filtered_pure_dep_tfed = []
            for i in range(nsims_cmb):
                if i % 10 == 0:
                    print("   ", i)
                mp = sp.load_cmb_sim(i, filtered=True)
                mp_masked = sp.get_masked_map(mp, nmt_purify=True,
                                              binary=False)
                mp_masked_dep, mp_template = ut.deproject_many(mp_masked,
                                                               mp_sims,
                                                               mat)
                cl = sp.compute_pspec(
                    mp_masked_dep,
                    transfer=transfer_pure_dep["full_tf"],
                    nmt_purify=False,
                    nmt_purify_mcm=True,
                    masked_on_input=True
                )
                cls_filtered_pure_dep_tfed.append(cl)

            cls_filtered_pure_dep_tfed = np.array(cls_filtered_pure_dep_tfed)
            np.savez(fname, cls=cls_filtered_pure_dep_tfed)

    # Plotting
    print("  5. Plotting")
    os.makedirs(f"{out_dir}/plots", exist_ok=True)

    plt.figure()
    plt.title('Cl, no filtering')

    cl2dl = sp.leff*(sp.leff+1)/2./np.pi
    y = np.mean(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)*cl2dl
    yerr = np.std(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)*cl2dl
    plt.plot(sp.leff, y, color='b', ls="-", label="Masked CMB, B only")
    plt.fill_between(sp.leff, y-yerr, y+yerr, color='b', alpha=0.2)

    y = np.mean(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)*cl2dl
    yerr = np.std(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)*cl2dl  # noqa: E501
    plt.plot(sp.leff, y, color='k', ls="-", label="Masked CMB, no purification")  # noqa: E501
    plt.fill_between(sp.leff, y-yerr, y+yerr, color='k', alpha=0.2)

    y = np.mean(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)*cl2dl
    yerr = np.std(np.array([cl["BB"] for cl in cls_masked_pure]), axis=0)*cl2dl
    plt.plot(sp.leff, y, color='y', ls="-", label="Masked CMB, purified")
    plt.fill_between(sp.leff, y-yerr, y+yerr, color='y', alpha=0.2)
    ls = np.arange(sp.lmax+1)
    plt.plot(clth["BB"][:sp.lmax+1]*ls*(ls+1)/2./np.pi, 'r--', alpha=0.5, label="Theory")  # noqa: E501

    plt.xlim([2, lmax_plot])
    plt.ylim((1e-6, 1e0))
    plt.xlabel(r"$\ell$", fontsize=14)
    plt.ylabel(r"$D_\ell^{BB}$", fontsize=14)
    plt.yscale('log')
    plt.legend()
    print(f"  PLOT SAVED {plot_dir}/cl_masked_BB.pdf")
    plt.savefig(f"{plot_dir}/cl_masked_BB.pdf", bbox_inches="tight")
    plt.close()

    thbb = sp.bins.bin_cell(clth["BB"][:sp.bins.lmax+1])
    y = np.mean(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)*cl2dl
    yerr = np.std(np.array([cl["BB"] for cl in cls_noe_masked]), axis=0)*cl2dl
    plt.plot(sp.leff, (y - thbb)/(yerr/np.sqrt(nsims_cmb)), 'b-', alpha=0.5, label="Masked CMB, B only")  # noqa: E501

    thbb = clth_msk_nopure["BB"]
    y = np.mean(np.array([cl["BB"] for cl in cls_masked_nopure]), axis=0)*cl2dl
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
            plt.plot(clth[pols][:sp.lmax+1]*ls*(ls+1)/2./np.pi, 'r--', alpha=0.5, label="Theory")  # noqa: E501

            plt.xlim([2, lmax_plot])
            plt.xlabel(r"$\ell$", fontsize=14)
            plt.ylabel(fr"$D_\ell^{{{pols}}}$", fontsize=14)
            if pols == "BB":
                plt.yscale('log')
                plt.ylim((1e-6, 2e-3))
            elif pols == "EE":
                plt.yscale('log')
                plt.ylim((1e-4, 1e0))
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
            plt.ylabel(fr"$\sigma(D_\ell^{{{pols}, X}})$", fontsize=14)
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
            plt.plot(clth[pols][:sp.lmax+1]*ls*(ls+1)/2./np.pi, 'r--', alpha=0.5, label="Theory")  # noqa: E501

            plt.xlim([2, lmax_plot])
            if pols == "BB":
                plt.yscale('log')
                plt.ylim((1e-6, 2e-3))
            elif pols == "EE":
                plt.yscale('log')
                plt.ylim((1e-4, 1e0))
            plt.xlabel(r"$\ell$", fontsize=14)
            plt.ylabel(fr"$D_\ell^{{{pols}}}$", fontsize=14)
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
