import utils as ut
from pixell import enmap
import numpy as np
import healpy as hp
import pymaster as nmt


# Changelog:
# * 2025/12/17: Open deprojection lmax parameter

class SimPure:
    def __init__(self, pix_type, base_dir, out_dir, filter_setup,
                 mask_filename, car_template=None, nside=None,
                 beam_fwhm=30.):
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
        self.bins = nmt.NmtBin.from_lmax_linear(self.lmax, 10)
        self.leff = self.bins.get_effective_ells()

        # general
        self.base_dir = base_dir  # "/pscratch/sd/k/kwolz/bbdev/simpure"
        self.filter_setup = filter_setup

    def get_masked_map(self, mp, nmt_purify=False, binary=False):
        """
        """
        assert mp.shape[0] in [2, 3]
        mask = self.mask_bin if binary else self.mask

        if not nmt_purify:
            if self.pix_type == "hp":
                return mp*mask
            else:
                return enmap.ndmap(mp*mask[None, :, :], wcs=self.wcs)
        else:
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
        sim_fname = f"cmb{pols_keep}_{res_str}_fwhm{self.beam_fwhm}.0_sim{sim_id:04d}_{self.pix_type.upper()}.fits"  # noqa: E501

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
        sim_fname = f"{typ}_{res_str}_fwhm{self.beam_fwhm}.0_sim{sim_id:04d}_{self.pix_type.upper()}.fits"  # noqa: E501

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
                         nmt_purify=False,
                         return_bp_win=False,
                         overwrite=False):
        """
        """
        tf_correct = transfer is not None
        pix_type = "hp" if self.wcs is None else "car"
        pure_label = "_nmt_purify" if nmt_purify else ""
        tf_label = "_tf_correct" if tf_correct else ""
        fn = f"{self.out_dir}/inv_coupling{tf_label}{pure_label}_{pix_type}.npz"  # noqa: E501

        return ut.get_inv_coupling(
            fn, self.mask, self.bins, transfer=transfer, nmt_purify=nmt_purify,
            return_bp_win=return_bp_win, wcs=self.wcs, overwrite=overwrite
        )

    def compute_pspec(self, map,
                      transfer=None,
                      nmt_purify=False,
                      return_just_field=False,
                      nmt_purify_mcm=None,
                      masked_on_input=False):
        """
        """
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
