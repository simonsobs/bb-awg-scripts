import numpy as np
import sqlite3
import os
from bundling_utils import read_map, coadd_maps
from coordinator import BundleCoordinator
import re


class _Coadder:
    def __init__(self, atomic_db, bundle_db, freq_channel, split_label="full", wafer=None,
                 pix_type="hp"):
        """
        Constructor for the _Coadder class. Reads in map information from
        atomic_db and bundling information from bundle_db.

        Parameters
        ----------
        atomic_db: str
            Path to the atomic database.
        bundle_db: str
            Path to the bundling database.
        freq_channel: str
            Frequency channel label indicating the frequency to be read from.
            For SAT1 and SAT3, possible choices are "f090", "f150".
        wafer: str
            Optional; label indicating the telescope wafer to include.
            If no wafer is provided, coadd maps made for all the wafers.
        pix_type : str
            Pixelization type. Admissible values are "hp", "car.
        """
        assert pix_type in ["hp", "car"]
        self.pix_type = pix_type
        self.fields_hp = range(3) if pix_type == "hp" else None

        self.atomic_db = atomic_db
        self.bundle_db = bundle_db
        self.freq_channel = freq_channel
        self.wafer = wafer

    def _get_obs_ids(self, bundle_id, null_prop_val=None):
        """
        Infer all obs_ids from the bundle_db given a bundle_id and a set of
        null properties.

        Parameters
        ----------
            bundle_id: int
                ID corresponding to the bundle that observations belong to.
            null_prop_val: str
                String of format "{quality}_{null_property}", e.g. "low_pwv"
                indicating the null split that observations belong to.

        Returns
        -------
            obs_ids: list
                List of obs_ids with the selected properties.
        """
        bundle_coord = BundleCoordinator.from_dbfile(
            self.bundle_db,
            bundle_id=bundle_id,
            null_prop_val=null_prop_val
        )
        obs_ids = bundle_coord.obs_id
        return obs_ids

    def _obsid2fnames(self, obs_id, return_weights=False, split_label="full"):
        """
        Given an obs_id, infer the file names of the corresponding atomic maps.

        Parameters
        ----------
            obs_id: str
                Observation ID to query after.
            return_weights: bool
                Whether to return mean QU polarization weights, needed for
                generating sign-flip noise realizations.

        Returns
        -------
            fnames: list
                List of strings with file paths.
        """
        map_dir = os.path.dirname(self.atomic_db)
        #print(split_label)
        # For now, assume HEAPix maps are gzipped fits and CAR maps are fits.
        # This may be generalized at some point in the future.
        suffix = ".fits.gz" if self.pix_type == "hp" else ".fits"

        con = sqlite3.connect(self.atomic_db)
        cursor = con.cursor()

        if return_weights:
            subquery = "ctime, wafer, freq_channel, median_weight_qu"
        else:
            subquery = "ctime, wafer, freq_channel"

        query = f"SELECT {subquery} FROM atomic WHERE freq_channel = "
        query += f"'{self.freq_channel}' AND obs_id = '{obs_id}'"
        if self.wafer is not None:
            query += f" AND wafer = '{self.wafer}'"
        result = cursor.execute(query).fetchall()
        con.close()
        if return_weights:
            fnames = [
                os.path.join(
                    map_dir, f"{str(ctime)[:5]}",
                    f"atomic_{ctime}_{wafer}_{freq_channel}_{split_label}_wmap{suffix}"
                )
                for ctime, wafer, freq_channel, _ in result
            ]
            weights = [weight for _, _, _, weight in result]
            return fnames, weights
        else:
            fnames = [
                os.path.join(
                    map_dir, f"{str(ctime)[:5]}",
                    f"atomic_{ctime}_{wafer}_{freq_channel}_{split_label}_wmap{suffix}"
                )
                for ctime, wafer, freq_channel in result
            ]
            return fnames

    def _check_maps_exist(self, fname, check_hits=True):
        """
        Check if atomic maps, including weights and hits, exist on disk.

        Parameters
        ----------
            fname: str
                File name corresponding to the weighted atomic map.
            check_hits: bool
                Whether to check if the hits file also exists on disk.

        Returns
        -------
            maps_exist: bool
                Whether all maps exist on disk.
        """
        maps_exist = (os.path.isfile(fname)
                      and os.path.isfile(fname.replace("wmap", "weights")))
        if check_hits:
            hits_exist = os.path.isfile(fname.replace("wmap", "hits"))
            maps_exist = hits_exist and maps_exist
        return maps_exist

    def _get_fnames(self, bundle_id, null_prop_val=None, split_label="full", return_weights=False):
        """
        Return file names (and, optionally, mean polarization weights) given a
        bundle_id and a null property.

        Parameters
        ----------
        bundle_id: int
            ID corresponding to the bundle that observations belong to.
        null_prop_val: str
            String of format "{quality}_{null_property}", e.g. "low_pwv"
            indicating the null split that observations belong to.
        return_weights: bool
            Whether to return mean QU polarization weights, needed for
            generating sign-flip noise realizations.

        Returns
        -------
        fnames: list
            List of strings with file paths.
        """
        obs_ids = self._get_obs_ids(bundle_id, null_prop_val)
        #print(obs_ids)
        fnames = []
        if return_weights:
            weights = []

        for obs_id in obs_ids:
            if return_weights:
                fname_list, weight_list = self._obsid2fnames(
                    obs_id, return_weights=return_weights, split_label=split_label
                )
                for fname, weight in zip(fname_list, weight_list):
                    if self._check_maps_exist(fname):
                        fnames.append(fname)
                        weights.append(weight)
            else:
                fname_list = self._obsid2fnames(
                    obs_id, return_weights=return_weights, split_label=split_label
                )
                for fname in fname_list:
                    if self._check_maps_exist(fname):
                        fnames.append(fname)
        if return_weights:
            return fnames, weights
        return fnames #, split_label


class Bundler(_Coadder):
    """
    Child class of _Coadder, with the purpose of coadding atomic maps for the
    purpose of generating map bundles and bundled hits maps.
    """
    def get_abscal_factors(self):
        calibration_factors = {
            'ws0': {'f090': 14.6, 'f150': 10.2},
            'ws1': {'f090': 12.8, 'f150': 12.2},
            'ws2': {'f090': 13.6, 'f150': 9.6},
            'ws3': {'f090': 13.6, 'f150': 10.4},
            'ws4': {'f090': 13.6, 'f150': 14.5},
            'ws5': {'f090': 11.2, 'f150': 10.2},
            'ws6': {'f090': 17.1, 'f150': 12.7},
            }
        return calibration_factors

    def get_abfac(self, fname, abscal=True):
        self.abscal_factors = self.get_abscal_factors() if abscal else 1
        ws, freq = self.extract_ws_freq(fname)
        abfac = self.abscal_factors[ws][freq] * 1e6 if abscal else 1
        return abfac

    def extract_ws_freq(self, input_str):
        """
        Extract 'ws' and 'freq' from the input string.

        Parameters
        ----------
        input_str: str
            Input string containing 'ws' and 'freq' information.

        Returns
        -------
        ws: str
            Extracted 'ws' value.
        freq: str
            Extracted 'freq' value.
        """
        pattern = r'ws\d+|f\d+'
        matches = re.findall(pattern, input_str)
        ws = next((match for match in matches if match.startswith('ws')), None)
        freq = next((match for match in matches if match.startswith('f')), None)
        return ws, freq

    def bundle(self, bundle_id, split_label, null_prop_val=None, abscal=True):
        """
        Make a map bundle given a bundle ID and, optionally, null properties.

        Parameters
        ----------
        bundle_id: int
            ID corresponding to the bundle that observations belong to.
        null_prop_val: str
            String of format "{quality}_{null_property}", e.g. "low_pwv"
            indicating the null split that observations belong to.

        Returns
        -------
        signal: np.array
            Output bundled signal map.
        hits: np.array
            Output bundled hits map.
        """
        fnames = self._get_fnames(bundle_id, null_prop_val, split_label)
        wmaps_list = [read_map(fname, pix_type=self.pix_type,
                               fields_hp=self.fields_hp) \
                                * self.get_abfac(fname, abscal)
                      for fname in fnames]
        weights_list = [read_map(fname.replace("wmap", "weights"),
                                 pix_type=self.pix_type,
                                 fields_hp=self.fields_hp,
                                 is_weights=True)
                        for fname in fnames]
        hits_list = [read_map(fname.replace("wmap", "hits"),
                              pix_type=self.pix_type)
                     for fname in fnames]
        signal, hits = coadd_maps(wmaps_list, weights_list, hits_list,
                                  pix_type=self.pix_type)

        return signal, hits


class SignFlipper(_Coadder):
    """
    Child class of _Coadder, with the purpose of sign-flipping and coadding
    atomic maps for the purpose of generating per-bundle noise maps.
    """
    def __init__(self, atomic_db, bundle_db, freq_channel, wafer=None,
                 bundle_id=None, null_prop_val=None, pix_type="hp"):
        """
        Constructor for the SignFlipper class. Creates a SignFlipper object,
        given map information from atomic_db and bundling information from
        bundle_db.

        Parameters
        ----------
        atomic_db: str
            Path to the atomic database.
        bundle_db: str
            Path to the bundling database.
        freq_channel: str
            Frequency channel label indicating the frequency to be read from.
            For SAT1 and SAT3, possible choices are "f090", "f150".
        wafer: str
            Optional; label indicating the telescope wafer to include.
            If no wafer is provided, coadd maps made for all the wafers.
        bundle_id: int
            Optional; ID corresponding to the bundle that observations belong
            to. If None, coadd maps from all bundles.
        null_prop_val: str
            String of format "{quality}_{null_property}", e.g. "low_pwv"
            indicating the null split that observations belong to.
        pix_type : str
            Pixelization type. Admissible values are "hp", "car.
        """
        super().__init__(atomic_db, bundle_db, freq_channel, wafer,
                         pix_type=pix_type)

        if self.pix_type == "hp":
            # Erik's atomic.db contains mean QU weights, so we'll use them here
            self.fnames, self.ws = self._get_fnames(
                bundle_id, null_prop_val, return_weights=True
            )
        else:
            self.fnames = self._get_fnames(
                bundle_id, null_prop_val, return_weights=False
            )

        self.wmaps = [read_map(fname, pix_type=self.pix_type,
                               fields_hp=self.fields_hp)
                      for fname in self.fnames]
        self.weights = [read_map(fname.replace("wmap", "weights"),
                                 pix_type=self.pix_type,
                                 fields_hp=self.fields_hp,
                                 is_weights=True)
                        for fname in self.fnames]

        if self.pix_type == "car":
            # Susanna's atomics.db doesn't have mean QU weights, so we'll
            # compute them by hand
            self.ws = [(np.mean(w[1]) + np.mean(w[2])) / 2
                       for w in self.weights]

    def signflip(self, seed=None):
        """
        Creates a signflip noise realization.

        Parameters
        ----------
        seed: int
            Random seed allowing to deterministically reproduce the noise
            realization.

        Returns
        -------
        signflip_noise: np.array
            TQU map with th sign-flip noise realization.
        """
        if seed is not None:
            np.random.seed(seed)
        perm_idx = np.random.permutation(len(self.wmaps))
        maps_list, weights_list = ([self.wmaps[i] for i in perm_idx],
                                   [self.weights[i] for i in perm_idx])
        mean_wQU_list = [self.ws[i] for i in perm_idx]

        weight_cumsum = np.cumsum(mean_wQU_list) / np.sum(mean_wQU_list)
        pm_one = np.random.choice([-1, 1])
        sign_list = np.where(weight_cumsum < 0.5, pm_one, -pm_one)

        signflip_noise = coadd_maps(maps_list, weights_list,
                                    pix_type=self.pix_type,
                                    sign_list=sign_list)

        return signflip_noise
