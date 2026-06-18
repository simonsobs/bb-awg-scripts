import utils as ut
from pixell import enmap
import numpy as np
import healpy as hp
import pymaster as nmt
import os
import sys
import matplotlib.pyplot as plt

# TODO: Make it an actual module
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'misc'))
)
import mpi_utils as mpi


# Changelog:
# * 2025/12/17: Open deprojection lmax parameter
# * 2026/01/25: ported functions to simpure
# * 2026/01/29: added flag tf_type to get_inv_coupling and compute_pspec
# * 2026/02/05: adapted filtered map naming convention for CAR
# * 2026/03/03: added MPI parallelization
# * 2026/05/05: added power-law validation
# * 2026/06/18: add documentation

class SimPure:
    def __init__(self, pix_type, base_dir, out_dir, filter_setup,
                 mask_filename, car_template=None, nside=None,
                 beam_fwhm=30., nlb=10):
        """
        Class for simulation-based purification.

        Arguments
        ---------
        :param pix_type: str
            Pixelization type. Supported are "car" and "hp".
        :param base_dir: str
            Base input directory for sim-based purification
        :param out_dir: array-like
            Output directory for sim-based purification
        :param filter_setup: str
            Name given to the filtering setup. Can be anything.
        :param mask_filename: str
            Full path to the (apodized) analysis mask
        :param car_template: str
            Full path to the CAR pixelization template. Ignored if
            pixelization_type is "hp".
        :beam_fwhm: float
            Beam resolution FWHM. Only used to load specific maps from disk.
        :nlb: int
            Number of multipoles to bin into a bandpower bin. Determines
            (linear) NaMaster bandpower binning scheme.
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

        # MPI related initialization
        self.rank, self.size, self.comm = mpi.init(True)

    def get_masked_map(self, mp, nmt_purify=False, binary=False):
        """
        Outputs map multiplied by the apodized mask.

        Arguments
        ---------
        :param mp: array-like
            Input (unmasked) TQU map
        :param nmt_purify: bool
            Whether to apply KS purification on the map level after masking
        :param binary: bool
            Whether to multiply by binary msk instead of apodized mask.
            Ignored if nmt_purify is False.
        
        Returns
        -------
        array-like
            Output map
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

    def load_val_sim(self, sim_id, filtered=False, pols_keep="EB", typ="cmb"):
        """
        Load validation simulation for purification.

        Arguments
        ---------
        :param sim_id: int
            Simulation ID to load
        :param filtered: bool
            Whether to load filtered simulation
        :param pols_keep: str
            What polarizations to keep in the map. Can be "EB" or "B".
        :param typ: str
            What type of validation map to load. Can be "cmb" or "plaw".

        Returns
        -------
        array-like
            Binary-masked TQU map, in muK units
        """
        sim_dir = f"filtered_{typ}_sims/{self.filter_setup}"  # obsmat, toy
        # sim_dir = f"filtered_{typ_lab}_sims/satp3/f090/{self.filter_setup}/coadded_sims"  # sotodlib  # noqa: E501
        if not filtered:
            sim_dir = {"cmb": "cmb_sims", "pure": "input_sims", "plaw": "plaw_sims"}[typ]  # noqa: E501
            pix_str = "CAR" if self.pix_type == "car" else "HP"
        else:
            pix_str = "CAR_f090_science_filtered" if self.pix_type == "car" else "HP"  # noqa: E501

        if (self.pix_type, typ) == ("car", "cmb"):
            res_str = "20arcmin"
        elif (self.pix_type, typ) == ("car", "plaw"):
            res_str = "20.0arcmin"
        else:
            res_str = f"nside{self.nside}"
        sim_fname = f"{typ}{pols_keep}_{res_str}_fwhm{self.beam_fwhm:.1f}_sim{sim_id:04d}_{pix_str}.fits"  # noqa: E501

        map = ut.read_map(
            f"{self.base_dir}/{sim_dir}/{sim_fname}",
            pix_type=self.pix_type,
            fields_hp=[0, 1, 2],
            car_template=self.car_template
        )
        conv = 1E6 if typ == "cmb" else 1. 
        return conv * self.get_masked_map(map, binary=True)

    def load_transfer_sim(self, sim_id, filtered=False, typ=None):
        """
        Load pure-type simulation for transfer function calculation.

        Arguments
        ---------
        :param sim_id: int
            Simulation ID to load
        :param filtered: bool
            Whether to load filtered simulation
        :param typ: str
            What type of validation map to load. Can be "pure{T|E|B}".

        Returns
        -------
        array-like
            Binary-masked TQU map
        """
        assert typ in [f"pure{p}" for p in "TEB"], "Invalid pure type"
        sim_dir = f"filtered_pure_sims/{self.filter_setup}"  # obsmat, toy
        # sim_dir = f"filtered_pure_sims/satp3/f090/{self.filter_setup}/coadded_sims"  # sotodlib
        if not filtered:
            sim_dir = "input_sims"
            pix_str = "CAR" if self.pix_type == "car" else "HP"
        else:
            pix_str = "CAR_f090_science_filtered" if self.pix_type == "car" else "HP"  # noqa: E501
        res_str = "20.0arcmin" if self.pix_type == "car" else f"nside{self.nside}"  # noqa: E501
        sim_fname = f"{typ}_{res_str}_fwhm{self.beam_fwhm:.1f}_sim{sim_id:04d}_{pix_str}.fits"  # noqa: E501

        map = ut.read_map(
            f"{self.base_dir}/{sim_dir}/{sim_fname}",
            pix_type=self.pix_type,
            fields_hp=[0, 1, 2],
            car_template=self.car_template
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
        Get inverse bandpower coupling matrix, accounting for mask-related mode
        coupling and potentially transfer function from other mask-based
        operations like filtering and purification.

        Arguments
        ---------
        :param transfer: array-like
            9x9xNbin transfer function
        :param tf_type: str
            Type of TF. Supported are None, "nopure", "pure", "pure_deproj",
            "pure_matrix".
        :param nmt_purify: bool
            Whether to account for KS purification on the mask-based mode
            coupling matrix.
        :return_bp_win: bool
            Whether to return the bandpower window function
        :param overwrite: bool
            Whether to overwrite existing inverse coupling files

        Returns
        -------
        :param inv_coupling: array-like
            9*Nbins x 9*Nbins inverse bandpower coupling matrix
        :param bp_win: array-like
            9 x nbins x 9 x nl bandpower window function 
        """
        pix_type = "hp" if self.wcs is None else "car"
        pure_label = "_nmt_purify" if nmt_purify else ""
        tf_label = {None: "", "nopure": "_tf_nopure",
                    "pure": "_tf_pure", "pure_deproj": "_tf_pure_deproj",
                    "pure_matrix": "_tf_pure_matrix"}[tf_type] 
        print("tf_label", tf_label)
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
        Compute power spectrum, or NaMaster field from an input map.

        Arguments
        ---------
        :param overwrite: map
            TQU input map. Assmed to be non-masked unless
            "masked_on_input" is set to True.
        :param transfer: array-like
            9x9xNbin transfer function
        :param tf_type: str
            Type of TF. Supported are None, "nopure", "pure", "pure_deproj",
            "pure_matrix".
        :param nmt_purify: bool
            Whether to apply KS purification on the map level
        :return_just_field: bool
            Whether to return NmtField object. If False (default), return
            decoupled power spectrum.
        :param nmt_purify_mcm: bool
            Whether to account for KS purification on the mask-based mode
            coupling matrix.
        :param masked_on_input: bool
            Whether assumes input map to have been multiplied by analysis mask
        
        Returns
        -------
        :param field: NmtField object
            Output field
        :param cl: dict of array-like
            Dictionary of decoupled power spectra, with keys "TT", "TE", etc.
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
        Extracts the pure-B component of a polarized map using harmonic
        transformation.

        Arguments
        ---------
        :param map: array-like
            Input TQU map
        
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

    def make_deprojection_matrix(self, sim_dir, sim_fn, deproj_mat_fn,
                                 nsims_purify,
                                 overwrite=False, mat_plot_fn=None):
        """
        Makes a deprojection matrix for simulation-based purification.

        Arguments
        ---------
        :param sim_dir: str
            Input directory for filtered pure-E simulations
        :param sim_fn: str
            File name for filtered pure-E simulations. Must contain fstring
            'sim_id'
        :param deproj_mat_fn: str
            Full file path to npz file that contains deprojection matrix. Will
            be written to if nonexistent or overwrite == True.
        :param nsims_purify: int
            Number of pure-E simulations to use to build deprojection matrix
        :param overwrite: bool
            Whether to overwrite deprojection sims and matrix

        Returns
        -------
        :param deproj_sims: array-like
            Array of E-to-B leakage template maps for deprojection
        :param deproj_mat: array-like
            Deprojection matrix (outer product of deproj_sims with itself,
            and cropped in dimensionality)
        """
        # Map B-residuals (B-only component of filtered,
        # mask-purified pure-E simulations)
        os.makedirs(sim_dir, exist_ok=True)
        pix_type = "hp" if self.wcs is None else "car"

        # First make deprojection templates
        mpi_shared_list = list(range(nsims_purify))
        mpi_shared_list = self.comm.bcast(mpi_shared_list, root=0)
        task_ids = mpi.distribute_tasks(self.size, self.rank,
                                        len(mpi_shared_list))
        local_mpi_list = [mpi_shared_list[i] for i in task_ids]
        print("rank", self.rank, local_mpi_list)

        for i in local_mpi_list:
            if i % 50 == 0:
                print("   MAKE TEMPLATE", i)
            fname = f"{sim_dir}/{sim_fn.format(id_sim=i)}"
            if os.path.isfile(fname) and not overwrite:
                continue
            else:
                try:
                    # Load filtered pure-E sim
                    mp = self.load_purification_sim(i, filtered=True)
                except FileNotFoundError:
                    print(f"Error loading purification sim {i}. Skipping.")
                    continue
                # Mask-purify
                mp_masked = self.get_masked_map(mp, nmt_purify=True,
                                                binary=False)
                # Extract B-modes and remap them
                mp_masked_bonly = self.extract_pure_mode(
                    mp_masked, self.lmax, "B")[-2:]
                ut.write_map(fname, mp_masked_bonly, pix_type=pix_type)
        self.comm.barrier()
        
        if self.rank == 0:
            # Then save the deprojection matrix
            deproj_sims = []
            for i in range(nsims_purify):
                if i % 50 == 0:
                    print("   SAVE MP SIMS", i)
                fname = f"{sim_dir}/{sim_fn.format(id_sim=i)}"
                if os.path.isfile(fname):
                    mp_masked_bonly = ut.read_map(
                        fname,
                        pix_type=pix_type,
                        fields_hp=[0, 1],
                        car_template=self.car_template
                    )
                else:
                    print(f"WARNING: {fname} is not present.")
                    continue
                deproj_sims.append(mp_masked_bonly)

            # Save M_ij = s_ipn *s_jpn, where s is the simulation vector
            # of B-residuals
            if os.path.isfile(deproj_mat_fn) and not overwrite:
                pass
            deproj_mat = []
            for i, s in enumerate(deproj_sims):
                if i % 50 == 0:
                    print("   SAVE MAT", i)
                if pix_type == "hp":
                    deproj_mat.append(np.sum(np.array(deproj_sims)*s[None, :, :], axis=(1, 2)))  # noqa: E501
                else:
                    deproj_mat.append(np.sum(np.array(deproj_sims)*s[None, :, :, :],  # noqa: E501
                                      axis=(1, 2, 3)))
            deproj_mat = np.array(deproj_mat)
            np.savez(deproj_mat_fn, deproj_mat=deproj_mat, deproj_sims=deproj_sims)  # noqa: E501
            print("SAVED MAT", deproj_mat_fn, deproj_mat.shape)

            if mat_plot_fn is not None:
                # Visualize eigenvalues of M
                w, _ = np.linalg.eigh(deproj_mat)
                plt.figure()
                plt.plot(w[::-1])
                plt.yscale('log')
                plt.savefig(mat_plot_fn, bbox_inches="tight")
                print(f"    PLOT SAVED {mat_plot_fn}")
                plt.close()
        self.comm.barrier()
        deproj_mat, deproj_sims = [np.load(deproj_mat_fn, allow_pickle=True)[k]
                                   for k in ["deproj_mat", "deproj_sims"]]

        return deproj_sims, deproj_mat

    def get_tf_spectra(self, out_dir, nsims, id_sim_start=0, overwrite=False,
                       filtered=True, purified=True, method=None,
                       deproj_sims=None, deproj_mat=None, matpure=None):
        """
        Compute TF input power spectra for purification.

        Arguments
        ---------
        :param out_dir: str
            Output directory for validation sim spectra
        :param nsims: int
            Number of simulations to compute spectra of
        :param id_sim_start: int
            Simulation ID to start
        :param overwrite: bool
            Whether to recompute power spectra instead of loading from disk
        :param filtered: bool
            Whether to compute spectra from filtered maps
        :param method: str
            What method of purification to use. Supports "matrix", "deproj".
        :param deprojected: bool
            Whether to compute spectra from sim-based purified maps
        :param deproj_sims: array-like
            Array of E-to-B leakage template maps for deprojection
        :param deproj_mat: array-like
            Deprojection matrix (outer product of deproj_sims with itself,
            and cropped in dimensionality)
        :param matpure: MatrixPurification object
            Class instance to perform matrix-based purification
        
        Returns
        -------

        array-like
            Array of binned coupled power spectra from TF simulations
        """
        pure_lab = {True: "pure", False: "nopure"}
        filt_lab = {True: "filtered", False: "unfiltered"}
        adv_lab = f"_{method}" if method is not None else ""
        fp = (filtered, purified)
        fname = f"{out_dir}/cls_tf_{filt_lab[filtered]}_{pure_lab[purified]}{adv_lab}_nsims{nsims}.npz"  # noqa: E501

        if os.path.isfile(fname) and not overwrite:
            cls_tf = np.load(fname, allow_pickle=True)["cls"]
        else:
            cls_tf = []
            for i in range(id_sim_start, id_sim_start+nsims):
                if i % 10 == 0:
                    print("   TF SIMS ", i)
                fields, fields2 = ({}, {})
                for typ in [f"pure{f}" for f in "TEB"]:
                    if fp == (False, False):
                        mp = self.load_transfer_sim(i, filtered=False, typ=typ)
                        f = self.compute_pspec(
                            mp, nmt_purify=False, return_just_field=True
                        )
                    elif fp == (False, True):
                        mp = self.load_transfer_sim(i, filtered=False, typ=typ)
                        f = self.compute_pspec(
                            mp, nmt_purify=True, return_just_field=True
                        )
                    elif fp == (True, False):
                        mp = self.load_transfer_sim(i, filtered=True, typ=typ)
                        f = self.compute_pspec(
                            mp, nmt_purify=False, return_just_field=True
                        )
                    elif fp == (True, True):
                        mp = self.load_transfer_sim(i, filtered=True, typ=typ)
                        if method is None:
                            f = self.compute_pspec(
                                mp, nmt_purify=True, return_just_field=True)
                        elif method == "matrix":
                            mp_masked = self.get_masked_map(mp,
                                                            nmt_purify=False,
                                                            binary=False)
                            mp_masked_pure = matpure.purify_observed_map(
                                mp_masked)
                            f = self.compute_pspec(
                                mp_masked_pure,
                                return_just_field=True,
                                masked_on_input=True
                            )
                        elif method == "deproj":
                            mp_masked = self.get_masked_map(mp,
                                                            nmt_purify=True,
                                                            binary=False)
                            mp_masked_pure, _ = ut.deproject_many(mp_masked,
                                                                  deproj_sims,
                                                                  deproj_mat)
                            f = self.compute_pspec(
                                mp_masked_pure,
                                return_just_field=True,
                                masked_on_input=True
                            )
                        else:
                            raise ValueError("Unknown purification method."
                                             "Choose either 'matrix', 'deproj'"
                                             ", or None.")
                    fields[typ] = f
                    fields2[typ] = f
                pcls_mat = self.get_pcls_mat_transfer(fields, fields2)
                cls_tf.append(pcls_mat)
            cls_tf = np.array(cls_tf)
            np.savez(fname, cls=cls_tf)
            print("SAVED TF SIMS", fname)
        return cls_tf

    def get_val_spectra(self,
                        out_dir,
                        nsims,
                        id_sim_start=0,
                        overwrite=False,
                        plot_dir=None,
                        map="cmb",
                        noE=False,
                        filtered=False,
                        purified=False,
                        method=None,
                        TFed=False,
                        deproj_sims=None,
                        deproj_mat=None,
                        matpure=None,
                        tf=None):
        """
        Compute validation spectra for purification.

        Arguments
        ---------
        :param out_dir: str
            Output directory for validation sim spectra
        :param nsims: int
            Number of simulations to compute spectra of
        :param id_sim_start: int
            Simulation ID to start
        :param overwrite: bool
            Whether to recompute power spectra instead of loading from disk
        :param plot_dir: str
            Plotting directory
        :param map: str
            Type of map to be used for validation. Can be "cmb" or "plaw".
        :param noE: bool
            Whether to compute spectra from B-mode-only maps
        :param filtered: bool
            Whether to compute spectra from filtered maps
        :param purified: bool
            Whether to apply NaMaster purification ot the maps
        :param method: str
            What method of advanced purification to use.
            Supported are "matrix", "deproj".
        :param TFed: bool
            Whether the power spectra have been transfer-fucntion corrected
        :param deprojected: bool
            Whether to compute spectra from sim-based purified maps
        :param deproj_sims: array-like
            Array of E-to-B leakage template maps
        :param deproj_mat: array-like
            Deprojection matrix (outer product of deproj_sims with itself,
            and cropped in dimensionality)
        :param matpure: MatrixPurification object
            Class instance to perform matrix-based purification
        :param tf: array-like
            9x9xNbin transfer function
        
        Returns
        -------
        dict of array-like
            Dictionary of decoupled power spectra, with keys "TT", "TE", etc.
        """
        nfpmt = (noE, filtered, purified, method, TFed)
        clabs = {
            (False, False, False, None, False): "masked_nopure",
            (False, False, True, None, False): "masked_pure",
            (True, False, False, None, False): "noe_masked",
            (False, True, False, None, False): "filtered_nopure",
            (False, True, False, None, True): "filtered_nopure_tfed",
            (False, True, True, None, False): "filtered_pure",
            (False, True, True, None, True): "filtered_pure_tfed",
            (True, True, False, None, False): "noe_filtered",
            (True, True, False, None, True): "noe_filtered_tfed",
            (True, True, False, None, False): "noe_filtered",
            (False, True, True, "matrix", False): "filtered_pure_matrix",
            (False, True, True, "matrix", True): "filtered_pure_matrix_tfed",
            (False, True, True, "deproj", False): "filtered_pure_deproj",
            (False, True, True, "deproj", True): "filtered_pure_deproj_tfed",
        }
        clab = clabs[nfpmt]
        if nfpmt not in clabs:
            raise ValueError("Your configuration is not supported:\n"
                             f"(noE {noE}, filtered {filtered}, "
                             f"purified {purified}, method {method}, "
                             f"TFed {TFed})")
        fname = f"{out_dir}/cls_{map}_{clab}.npz"
        if os.path.isfile(fname) and not overwrite:
            cls = np.load(fname, allow_pickle=True)['cls']
        else:
            cls = []
            for i in range(id_sim_start, id_sim_start+nsims):
                if i % 10 == 0:
                    print(f"   CELLS {map} ", i)
                if clab == "masked_nopure":
                    mp = self.load_val_sim(i, filtered=False, typ=map)
                    cl = self.compute_pspec(mp)
                elif clab == "masked_pure":
                    mp = self.load_val_sim(i, filtered=False, typ=map)
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
                    mp = self.load_val_sim(i, pols_keep="B", typ=map)
                    cl = self.compute_pspec(mp)
                elif clab == "filtered_nopure":
                    mp = self.load_val_sim(i, filtered=True, typ=map)
                    if i == 0:
                        ut.plot_map(
                            mp,
                            file_name=f"{plot_dir}/sim_{map}_filt_{i:04d}",
                            pix_type=self.pix_type
                        )
                        ut.plot_map(
                            self.get_masked_map(mp, nmt_purify=False, binary=False),  # noqa: E501
                            file_name=f"{plot_dir}/sim_{map}_filt_masked_{i:04d}",  # noqa: E501
                            pix_type=self.pix_type
                        )
                    cl = self.compute_pspec(mp, transfer=None, nmt_purify=False)  # noqa: E501
                elif clab == "filtered_nopure_tfed":
                    mp = self.load_val_sim(i, filtered=True, typ=map)
                    if i == 0:
                        ut.plot_map(
                            mp,
                            file_name=f"{plot_dir}/sim_{map}_filt_{i:04d}",
                            pix_type=self.pix_type
                        )
                        ut.plot_map(
                            self.get_masked_map(mp, nmt_purify=False, binary=False),  # noqa: E501
                            file_name=f"{plot_dir}/sim_{map}_filt_masked_{i:04d}",  # noqa: E501
                            pix_type=self.pix_type
                        )
                    cl = self.compute_pspec(
                        mp,
                        transfer=tf,
                        tf_type="nopure",
                        nmt_purify=False
                    )
                elif clab == "filtered_pure":
                    mp = self.load_val_sim(i, filtered=True, typ=map)
                    cl = self.compute_pspec(mp, transfer=None, nmt_purify=True)
                elif clab == "filtered_pure_tfed":
                    mp = self.load_val_sim(i, filtered=True, typ=map)
                    cl = self.compute_pspec(
                        mp,
                        transfer=tf,
                        tf_type="pure",
                        nmt_purify=True
                    )
                elif clab == "noe_filtered":
                    mp = self.load_val_sim(i, filtered=True, pols_keep="B",
                                           typ=map)
                    cl = self.compute_pspec(mp, transfer=None)
                elif clab == "noe_filtered_tfed":
                    mp = self.load_val_sim(i, filtered=True, pols_keep="B",
                                           typ=map)
                    cl = self.compute_pspec(mp, transfer=tf, tf_type="nopure")
                elif "filtered_pure_deproj" in clab:
                    mp = self.load_val_sim(i, filtered=True, typ=map)
                    mp_masked = self.get_masked_map(mp, nmt_purify=True,
                                                    binary=False)
                    mp_masked_dep, _ = ut.deproject_many(mp_masked,
                                                         deproj_sims,
                                                         deproj_mat)
                    tf_type = None
                    if clab == "filtered_pure_deproj_tfed":
                        tf_type = "pure_deproj"
                    cl = self.compute_pspec(
                        mp_masked_dep,
                        transfer=tf,
                        tf_type=tf_type,
                        nmt_purify=False,
                        nmt_purify_mcm=True,
                        masked_on_input=True
                    )
                elif "filtered_pure_matrix" in clab:
                    mp = self.load_val_sim(i, filtered=True, typ=map)
                    mp_masked = self.get_masked_map(mp,
                                                    nmt_purify=False,
                                                    binary=False)
                    mp_masked_pure = matpure.purify_observed_map(mp_masked)
                    tf_type = None
                    if clab == "filtered_pure_matrix_tfed":
                        tf_type = "pure_matrix"
                    cl = self.compute_pspec(
                        mp_masked_pure,
                        transfer=tf,
                        tf_type=tf_type,
                        nmt_purify=False,
                        nmt_purify_mcm=False,
                        masked_on_input=True
                    )

                cls.append(cl)
            cls = np.array(cls)
            np.savez(fname, cls=cls)
        return cls
