import utils as ut
from pixell import enmap
import numpy as np
import healpy as hp
import pymaster as nmt
import os
import matplotlib.pyplot as plt


# Changelog:
# * 2025/12/17: Open deprojection lmax parameter
# * 2026/01/25: ported functions to simpure
# * 2026/01/29: added flag tf_type to get_inv_coupling and compute_pspec

class SimPure:
    def __init__(self, pix_type, base_dir, out_dir, filter_setup,
                 mask_filename, car_template=None, nside=None,
                 beam_fwhm=30., nlb=10):
        """
        """
        self.pix_type = pix_type
        if self.pix_type == "hp":
            if nside is None:
                raise ValueError("nside must be provided.")
            self.shape = hp.nside2npix(nside)
            self.nside = nside
            self.wcs = None
            self.lmax = 3*self.nside - 1
            self.car_template = None
        elif self.pix_type == "car":
            if car_template is None:
                raise ValueError("car_template must be provided.")
            self.car_template = car_template
            self.nside = None
            self.shape, self.wcs = enmap.read_map_geometry(car_template)
            self.lmax = int(np.pi/np.deg2rad(
                np.min(np.abs(self.wcs.wcs.cdelt)))
            )
        self.beam_fwhm = beam_fwhm
        self.out_dir = out_dir
        self.mask = ut.read_map(mask_filename, pix_type=self.pix_type,
                                car_template=self.car_template)
        self.mask_bin = (self.mask > 0).astype(float)
        if self.pix_type == "car":
            self.mask_bin = enmap.ndmap(self.mask_bin, wcs=self.wcs)

        if 180.*60/self.lmax > beam_fwhm:
            print("WARNING: beam resolution is lower than pixel scale. "
                  "Consider increasing the beam FWHM.")

        # binning
        self.bins = nmt.NmtBin.from_lmax_linear(self.lmax, nlb)
        self.leff = self.bins.get_effective_ells()

        # general
        self.base_dir = base_dir  # "/pscratch/sd/k/kwolz/bbdev/simpure"
        self.filter_setup = filter_setup

    def get_masked_map(self, mp, nmt_purify=False, binary=False):
        """
        binary=True is ignored if nmt_purify is False.
        """
        assert mp.shape[0] in [2, 3]

        if not nmt_purify:
            mask = self.mask_bin if binary else self.mask
            if self.pix_type == "hp":
                return mp*mask
            else:
                return enmap.ndmap(mp*mask[None, :, :], wcs=self.wcs)
        else:
            if binary:
                print("WARNING: Set binary=False because purification is on.")
            mask = self.mask
            f = nmt.NmtField(mask, mp[-2:], purify_b=True, wcs=self.wcs)
            mp_masked = np.array([m.reshape(self.shape) for m in f.get_maps()])
            if mp.shape[0] == 3:  # TQU maps
                mp_masked = np.array([mp[0]*mask,
                                      mp_masked.copy()[0],
                                      mp_masked.copy()[1]])
            if self.pix_type == "hp":
                return mp_masked
            else:
                # f.get_maps() doesn't preserve the wcs convention
                mp_masked = np.flip(mp_masked, axis=(1, 2))
                return enmap.ndmap(mp_masked, wcs=self.wcs)

    def load_cmb_sim(self, sim_id, filtered=False, pols_keep="EB"):
        """
        """
        sim_dir = f"filtered_cmb_sims/{self.filter_setup}"
        if not filtered:
            sim_dir = "cmb_sims"
        # TODO: generalize
        res_str = "20arcmin" if self.pix_type == "car" else f"nside{self.nside}"  # noqa: E501
        sim_fname = f"cmb{pols_keep}_{res_str}_fwhm{self.beam_fwhm:.1f}_sim{sim_id:04d}_{self.pix_type.upper()}.fits"  # noqa: E501

        map = ut.read_map(
            f"{self.base_dir}/{sim_dir}/{sim_fname}",
            pix_type=self.pix_type,
            fields_hp=[0, 1, 2],
            car_template=self.car_template,
            convert_K_to_muK=True
        )
        return self.get_masked_map(map, binary=True)

    def load_transfer_sim(self, sim_id, filtered=False, typ=None):
        """
        """
        assert typ in [f"pure{p}" for p in "TEB"], "Invalid pure type"
        sim_dir = f"filtered_pure_sims/{self.filter_setup}"
        if not filtered:
            sim_dir = "input_sims"
        res_str = "20.0arcmin" if self.pix_type == "car" else f"nside{self.nside}"  # noqa: E501
        sim_fname = f"{typ}_{res_str}_fwhm{self.beam_fwhm:.1f}_sim{sim_id:04d}_{self.pix_type.upper()}.fits"  # noqa: E501

        map = ut.read_map(
            f"{self.base_dir}/{sim_dir}/{sim_fname}",
            pix_type=self.pix_type,
            fields_hp=[0, 1, 2],
            car_template=self.car_template,
            convert_K_to_muK=True
        )
        return self.get_masked_map(map, binary=True)

    def load_purification_sim(self, sim_id, filtered=False):
        return self.load_transfer_sim(sim_id, filtered=filtered, typ="pureE")

    def get_inv_coupling(self,
                         transfer=None,
                         tf_type=None,
                         nmt_purify=False,
                         return_bp_win=False,
                         overwrite=False):
        """
        """
        pix_type = "hp" if self.wcs is None else "car"
        pure_label = "_nmt_purify" if nmt_purify else ""
        tf_label = {None: "", "nopure": "_tf_nopure",
                    "pure": "_tf_pure", "pure_dep": "_tf_pure_dep"}[tf_type] 
        fn = f"{self.out_dir}/inv_coupling{tf_label}{pure_label}_{pix_type}.npz"  # noqa: E501

        return ut.get_inv_coupling(
            fn, self.mask, self.bins, transfer=transfer, nmt_purify=nmt_purify,
            return_bp_win=return_bp_win, wcs=self.wcs, overwrite=overwrite
        )

    def compute_pspec(self, map,
                      transfer=None,
                      tf_type=None,
                      nmt_purify=False,
                      return_just_field=False,
                      nmt_purify_mcm=None,
                      masked_on_input=False):
        """
        """
        assert map.shape[0] == 3
        lmax = self.bins.lmax
        field = {
            "spin0": nmt.NmtField(
                self.mask, map[:1], wcs=self.wcs, lmax=lmax,
                masked_on_input=masked_on_input
            ),
            "spin2": nmt.NmtField(
                self.mask, map[1:], wcs=self.wcs, purify_b=nmt_purify,
                lmax=lmax, masked_on_input=masked_on_input
            )
        }

        if return_just_field:
            return field

        pcls = ut.get_coupled_pseudo_cls(field, field, self.bins)
        if nmt_purify_mcm is None:
            nmt_purify_mcm = nmt_purify
        inv_coupling = self.get_inv_coupling(transfer=transfer,
                                             tf_type=tf_type,
                                             nmt_purify=nmt_purify_mcm)

        return ut.decouple_pseudo_cls(pcls, inv_coupling)

    def extract_pure_mode(self, map, lmax, which_mode="B"):
        """
        """
        alms = {}
        alms["T"], alms["E"], alms["B"] = ut.map2alm(map, lmax,
                                                     pix_type=self.pix_type)
        for p in "TEB":
            if p != which_mode:
                alms[p] *= 0.

        return ut.alm2map(
            [alms["T"], alms["E"], alms["B"]], pix_type=self.pix_type,
            nside=self.nside, car_map_template=self.car_template
        )

    def get_pcls_mat_transfer(self, fields, fields2=None):
        return ut.get_pcls_mat_transfer(fields, self.bins, fields2)

    def make_deprojection_matrix(self, sim_dir, sim_fn, mat_fn, nsims_purify,
                                 overwrite=False, mat_plot_fn=None):
        """
        Docstring for make_deprojection_matrix
        """
        # Map B-residuals (B-only component of filtered,
        # mask-purified pure-E simulations)
        os.makedirs(sim_dir, exist_ok=True)
        pix_type = "hp" if self.wcs is None else "car"

        mp_sims = []

        for i in range(nsims_purify):
            if i % 50 == 0:
                print("   ", i)
            fname = f"{sim_dir}/{sim_fn.format(id_sim=i)}"
            if os.path.isfile(fname) and not overwrite:
                mp_masked_bonly = ut.read_map(
                    fname,
                    pix_type=pix_type,
                    fields_hp=[0, 1],
                    car_template=self.car_template
                )
            else:
                # Load filtered pure-E sim
                mp = self.load_purification_sim(i, filtered=True)
                # Mask-purify
                mp_masked = self.get_masked_map(mp, nmt_purify=True,
                                                binary=False)
                # Extract B-modes and remap them
                mp_masked_bonly = self.extract_pure_mode(
                    mp_masked, 3*self.nside, "B")[-2:]
                ut.write_map(fname, mp_masked_bonly, pix_type=pix_type)
            mp_sims.append(mp_masked_bonly)
        mp_sims = np.array(mp_sims)

        # Save M_ij = s_ipn *s_jpn, where s is the simulation vector
        # of B-residuals
        if os.path.isfile(mat_fn) and not overwrite:
            mat = np.load(mat_fn, allow_pickle=True)['mat']
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
            np.savez(mat_fn, mat=mat)

        if mat_plot_fn is not None:
            # Visualize eigenvalues of M
            w, _ = np.linalg.eigh(mat)
            plt.plot(w[::-1])
            plt.yscale('log')
            plt.savefig(mat_plot_fn, bbox_inches="tight")
            print(f"    PLOT SAVED {mat_plot_fn}")
            plt.close()

        return mp_sims, mat

    def get_tf_sims(self, out_dir, nsims, id_sim_start=0, overwrite=False,
                    filtered=True, purified=True, deprojected=True,
                    mp_sims=None, mat=None):
        """
        Docstring for get_tf_sims

        :param self: Description
        :param overwrite: Description
        :param filtered: Description
        :param purified: Description
        :param deprojected: Description
        """
        pure_lab = {True: "pure", False: "nopure"}
        filt_lab = {True: "filtered", False: "unfiltered"}
        dep_lab = {True: "_dep", False: ""}
        fpd = (filtered, purified, deprojected)
        fname = f"{out_dir}/cls_tf_{filt_lab[filtered]}_{pure_lab[purified]}_{dep_lab[deprojected]}_nsims{nsims}.npz"  # noqa: E501
        if os.path.isfile(fname) and not overwrite:
            cls_tf = np.load(fname, allow_pickle=True)["cls"]
        else:
            cls_tf = []
            for i in range(id_sim_start, id_sim_start+nsims):
                if i % 10 == 0:
                    print("   ", i)
                fields, fields2 = ({}, {})
                for typ in [f"pure{f}" for f in "TEB"]:
                    if fpd == (False, False, False):
                        mp = self.load_transfer_sim(i, filtered=False, typ=typ)
                        f = self.compute_pspec(
                            mp, nmt_purify=False, return_just_field=True
                        )
                        fields[typ] = f
                        fields2[typ] = f
                    elif fpd == (False, True, False):
                        mp = self.load_transfer_sim(i, filtered=False, typ=typ)
                        f = self.compute_pspec(
                            mp, nmt_purify=True, return_just_field=True
                        )
                        fields[typ] = f
                        fields2[typ] = f
                        # DEBUG
                        hp.write_map(
                            f"{out_dir}/{typ}_map_unfilt_pure_isim{i}.fits",
                            self.get_masked_map(mp, nmt_purify=True,
                                                binary=False),
                            overwrite=True, dtype=float
                        )
                    elif fpd == (True, False, False):
                        mp = self.load_transfer_sim(i, filtered=True, typ=typ)
                        f = self.compute_pspec(
                            mp, nmt_purify=False, return_just_field=True
                        )
                    elif fpd == (True, True, False):
                        mp = self.load_transfer_sim(i, filtered=True, typ=typ)
                        f = self.compute_pspec(
                            mp, nmt_purify=True, return_just_field=True
                        )
                    elif fpd == (True, True, True):
                        mp = self.load_transfer_sim(i, filtered=True, typ=typ)
                        mp_masked = self.get_masked_map(mp, nmt_purify=True,
                                                        binary=False)
                        # DEBUG
                        hp.write_map(
                            f"{out_dir}/{typ}_map_filt_pure_isim{i}.fits",
                            mp_masked, overwrite=True, dtype=float
                        )
                        mp_masked_dep, _ = ut.deproject_many(mp_masked,
                                                             mp_sims,
                                                             mat)
                        # DEBUG
                        hp.write_map(
                            f"{out_dir}/{typ}_map_filt_dep_isim{i}.fits",
                            mp_masked_dep, overwrite=True, dtype=float
                        )
                        f = self.compute_pspec(
                            mp_masked_dep,
                            return_just_field=True,
                            masked_on_input=True
                        )
                    fields[typ] = f
                    fields2[typ] = f
                pcls_mat = self.get_pcls_mat_transfer(fields, fields2)
                cls_tf.append(pcls_mat)
            cls_tf = np.array(cls_tf)
            np.savez(fname, cls=cls_tf)
        return cls_tf

    def get_cmb_spectra(self,
                        out_dir,
                        nsims,
                        noE=False,
                        filtered=False,
                        purified=False,
                        deprojected=False,
                        TFed=False,
                        mp_sims=None,
                        mat=None,
                        tf=None,
                        overwrite=False,
                        plot_dir=None):
        """
        Docstring for get_cmb_spectra

        :param self: Description
        :param out_dir: Description
        :param nsims: Description
        :param noE: Description
        :param filtered: Description
        :param purified: Description
        :param deprojected: Description
        :param mp_sims: Description
        :param mat: Description
        :param tf: Description
        :param overwrite: Description
        :param plot_dir: Description
        """
        nfpdt = (noE, filtered, purified, deprojected, TFed)
        clab = {
            (False, False, False, False, False): "masked_nopure",
            (False, False, True, False, False): "masked_pure",
            (True, False, False, False, False): "noe_masked",
            (False, True, False, False, False): "filtered_nopure",
            (False, True, False, False, True): "filtered_nopure_tfed",
            (False, True, True, False, False): "filtered_pure",
            (False, True, True, False, True): "filtered_pure_tfed",
            (True, True, False, False, False): "noe_filtered",
            (True, True, False, False, True): "noe_filtered_tfed",
            (False, True, True, True, False): "filtered_pure_dep",
            (False, True, True, True, True): "filtered_pure_dep_tfed"
        }[nfpdt]
        fname = f"{out_dir}/cls_{clab}.npz"
        if os.path.isfile(fname) and not overwrite:
            cls = np.load(fname, allow_pickle=True)['cls']
        else:
            cls = []
            for i in range(nsims):
                if i % 10 == 0:
                    print("   ", i)
                if clab == "masked_nopure":
                    mp = self.load_cmb_sim(i, filtered=False)
                    cl = self.compute_pspec(mp)
                elif clab == "masked_pure":
                    mp = self.load_cmb_sim(i, filtered=False)
                    if i == 0:
                        ut.plot_map(
                            mp,
                            file_name=f"{plot_dir}/sim_cmb_{i:04d}",
                            pix_type=self.pix_type
                        )
                        ut.plot_map(
                            self.get_masked_map(mp, nmt_purify=False, binary=False),  # noqa: E501
                            file_name=f"{plot_dir}/sim_cmb_masked_{i:04d}",
                            pix_type=self.pix_type
                        )
                        ut.plot_map(
                            self.get_masked_map(mp, nmt_purify=True, binary=False),  # noqa: E501
                            file_name=f"{plot_dir}/sim_cmb_masked_purif_{i:04d}",  # noqa: E501
                            pix_type=self.pix_type
                        )
                    cl = self.compute_pspec(mp, nmt_purify=True)
                elif clab == "noe_masked":
                    mp = self.load_cmb_sim(i, pols_keep="B")
                    cl = self.compute_pspec(mp)
                elif clab == "filtered_nopure":
                    mp = self.load_cmb_sim(i, filtered=True)
                    if i == 0:
                        ut.plot_map(
                            mp,
                            file_name=f"{plot_dir}/sim_cmb_filt_{i:04d}",
                            pix_type=self.pix_type
                        )
                        ut.plot_map(
                            self.get_masked_map(mp, nmt_purify=False, binary=False),  # noqa: E501
                            file_name=f"{plot_dir}/sim_cmb_filt_masked_{i:04d}",  # noqa: E501
                            pix_type=self.pix_type
                        )
                    cl = self.compute_pspec(mp, transfer=None, nmt_purify=False)  # noqa: E501
                elif clab == "filtered_nopure_tfed":
                    mp = self.load_cmb_sim(i, filtered=True)
                    if i == 0:
                        ut.plot_map(
                            mp,
                            file_name=f"{plot_dir}/sim_cmb_filt_{i:04d}",
                            pix_type=self.pix_type
                        )
                        ut.plot_map(
                            self.get_masked_map(mp, nmt_purify=False, binary=False),  # noqa: E501
                            file_name=f"{plot_dir}/sim_cmb_filt_masked_{i:04d}",  # noqa: E501
                            pix_type=self.pix_type
                        )
                    cl = self.compute_pspec(
                        mp,
                        transfer=tf,
                        tf_type="nopure",
                        nmt_purify=False
                    )
                elif clab == "filtered_pure":
                    mp = self.load_cmb_sim(i, filtered=True)
                    cl = self.compute_pspec(mp, transfer=None, nmt_purify=True)
                elif clab == "filtered_pure_tfed":
                    mp = self.load_cmb_sim(i, filtered=True)
                    cl = self.compute_pspec(
                        mp,
                        transfer=tf,
                        tf_type="pure",
                        nmt_purify=True
                    )
                elif clab == "noe_filtered":
                    mp = self.load_cmb_sim(i, filtered=True, pols_keep="B")
                    cl = self.compute_pspec(mp, transfer=None)
                elif clab == "noe_filtered_tfed":
                    mp = self.load_cmb_sim(i, filtered=True, pols_keep="B")
                    cl = self.compute_pspec(mp, transfer=tf, tf_type="nopure")
                elif clab == "filtered_pure_dep":
                    mp = self.load_cmb_sim(i, filtered=True)
                    mp_masked = self.get_masked_map(mp, nmt_purify=True,
                                                    binary=False)
                    mp_masked_dep, mp_template = ut.deproject_many(mp_masked,
                                                                   mp_sims,
                                                                   mat)
                    if i == 1:
                        ut.plot_map(
                            mp_template,
                            file_name=f"{plot_dir}/sim_cmb_filt_template_{i:04d}",  # noqa: E501
                            pix_type=self.pix_type
                        )
                        ut.plot_map(
                            mp_masked,
                            file_name=f"{plot_dir}/sim_cmb_filt_purify_{i:04d}",  # noqa: E501
                            pix_type=self.pix_type
                        )
                        ut.plot_map(
                            mp_masked_dep,
                            file_name=f"{plot_dir}/sim_cmb_filt_purify_dep_{i:04d}",  # noqa: E501
                            pix_type=self.pix_type
                        )
                    cl = self.compute_pspec(
                        mp_masked_dep,
                        transfer=None,
                        nmt_purify=False,
                        nmt_purify_mcm=True,
                        masked_on_input=True
                    )
                elif clab == "filtered_pure_dep_tfed":
                    mp = self.load_cmb_sim(i, filtered=True)
                    mp_masked = self.get_masked_map(mp, nmt_purify=True,
                                                    binary=False)
                    mp_masked_dep, mp_template = ut.deproject_many(mp_masked,
                                                                   mp_sims,
                                                                   mat)
                    cl = self.compute_pspec(
                        mp_masked_dep,
                        transfer=tf,
                        tf_type="pure_dep",
                        nmt_purify=False,
                        nmt_purify_mcm=True,
                        masked_on_input=True
                    )

                cls.append(cl)
            cls = np.array(cls)
            np.savez(fname, cls=cls)
        return cls
