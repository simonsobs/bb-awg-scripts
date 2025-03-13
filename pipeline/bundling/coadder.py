import numpy as np
import sqlite3
import os
from bundling_utils import read_map, coadd_maps
from coordinator import BundleCoordinator
import re
from pixell import enmap


class _Coadder:
    def __init__(self, atomic_db, bundle_db, freq_channel, wafer=None,
                 pix_type="hp", atomic_list=None, car_map_template=None,
                 telescope=None, query_restrict=None):
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
        atomics_list: list
            External list of string tuples (obs_id, wafer, freq_channel) of
            atomic maps that are to be used for the bundling.
            All other atomics in atomic_db will be left out.
            Ignore if None (default).
        telescope: str
            Label for telescope
        """
        assert pix_type in ["hp", "car"]
        self.pix_type = pix_type
        self.fields_hp = range(3) if pix_type == "hp" else None

        self.atomic_db = atomic_db
        self.bundle_db = bundle_db
        self.freq_channel = freq_channel
        self.wafer = wafer
        self.atomic_list = atomic_list
        self.car_map_template = car_map_template
        self.query_restrict = query_restrict

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

        # # Restrict list of atomics in atomic_db
        # if self.atomic_list is not None:
        #     obs_ids_restricted = []
        #     for obs_id, _, _ in self.atomic_list:
        #         if obs_id in obs_ids:
        #             obs_ids_restricted.append(obs_id)
        #     obs_ids = obs_ids_restricted

        return obs_ids

    def _obsid2fnames(self, obs_id, return_weights=False, split_label=None, map_dir=None):
        """
        TODO: If an atomics_list is given (w/ entries (obs_id, wafer, freq)),
        select only fnames corresponding to these atomics.
        Make this and attribute of _Coadder, and call it in __init__.

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
        if split_label is None:
            split_label = "full"
        if map_dir is None:
            map_dir = os.path.dirname(self.atomic_db)

        # For now, assume HEAPix maps are gzipped fits and CAR maps are fits.
        # This may be generalized at some point in the future.
        suffix = ".fits.gz" if self.pix_type == "hp" else ".fits"

        con = sqlite3.connect(self.atomic_db)
        cursor = con.cursor()

        if return_weights:
            subquery = "ctime, wafer, freq_channel, prefix_path, median_weight_qu"
        else:
            subquery = "ctime, wafer, freq_channel, prefix_path"

        query = f"SELECT {subquery} FROM atomic WHERE freq_channel = "
        query += f"'{self.freq_channel}' AND obs_id = '{obs_id}'"
        if self.wafer is not None:
            query += f" AND wafer = '{self.wafer}'"

        # First we query the science splits to get wafer-level restricts on eg weight
        query1 = query + " AND split_label='science'"
        if self.query_restrict is not None:
            query1 += (" AND " + self.query_restrict)
        result1 = cursor.execute(query1).fetchall()
        wafer_freq = [r[1] + r[2] for r in result1]

        # Then we do the acutal query for valid splits
        query2 = query + " AND valid = 1"
        if split_label is not None:
            query2 += f" AND split_label = '{split_label}'"
        result2 = cursor.execute(query2).fetchall()
        # Filter for the good results from query1
        result = []
        for r in result2:
            if r[1] + r[2] in wafer_freq:
                result.append(r)

        # Restrict list of atomics in atomic_db
        if self.atomic_list is not None:
            obs_ids = [k[0] for k in self.atomic_list]
            if obs_id not in obs_ids:
                if return_weights:
                    return [], []
                else:
                    return []
            else:
                result_restricted = []
                for o, w, f in self.atomic_list:
                    for r in result:
                        if o == obs_id and w == r[1] and f == r[2]:
                            result_restricted.append(r)
                result = result_restricted

        con.close()
        if return_weights:
            fnames = [
                os.path.join(
                    map_dir, f"{str(ctime)[:5]}",
                    f"{os.path.basename(prefix_path)}_wmap{suffix}"  # noqa
                )
                for ctime, _, _, prefix_path, _ in result
            ]
            weights = [weight for _, _, _, _, weight in result]
            return fnames, weights
        else:
            fnames = [
                os.path.join(
                    map_dir, f"{str(ctime)[:5]}",
                    f"{os.path.basename(prefix_path)}_wmap{suffix}"  # noqa
                )
                for ctime, _, _, prefix_path in result
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

    def _get_fnames(self, bundle_id, null_prop_val=None, split_label=None,
                    return_weights=False, map_dir=None):
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
        fnames = []
        if return_weights:
            weights = []

        if (split_label is None) or isinstance(split_label, str):
            split_labels = [split_label]
        else:
            split_labels = split_label

        for obs_id in obs_ids:
            for split_label in split_labels:
                if return_weights:
                    fname_list, weight_list = self._obsid2fnames(
                        obs_id,
                        return_weights=return_weights,
                        split_label=split_label,
                        map_dir=map_dir
                    )
                    for fname, weight in zip(fname_list, weight_list):
                        if self._check_maps_exist(fname):
                            fnames.append(fname)
                            weights.append(weight)
                else:
                    fname_list = self._obsid2fnames(
                        obs_id,
                        return_weights=return_weights,
                        split_label=split_label,
                        map_dir=map_dir
                    )
                    for fname in fname_list:
                        if self._check_maps_exist(fname):
                            fnames.append(fname)
        if return_weights:
            return fnames, weights
        else:
            return fnames


class Bundler(_Coadder):
    """
    Child class of _Coadder, with the purpose of coadding atomic maps for the
    purpose of generating map bundles and bundled hits maps.
    """
    def get_abscal_factors(self):
        calibration_factors = {
        'satp1': {  # noqa
            'ws0': {'f090': 14.964962501118226, 'f150': 12.972798880555338},
            'ws1': {'f090': 13.213743485029925, 'f150': 8.759982749240976},
            'ws2': {'f090': 14.964962501118226, 'f150': 12.58841965446481},
            'ws3': {'f090': 13.355826748309815, 'f150': 10.176267385345804},
            'ws4': {'f090': 13.649361402118823, 'f150': 9.996686196192643},
            'ws5': {'f090': 138.01020973253478, 'f150': 15.591161957364674},
            'ws6': {'f090': 15.722682121428011, 'f150': 11.639977077758559},
            },
         'satp3': {
             'ws0': {'f090': 14.532475084835909, 'f150': 10.36656358545177},
             'ws1': {'f090': 12.793546442205974, 'f150': 12.40588756947507},
             'ws2': {'f090': 12.420918875928129, 'f150': 8.092555492155949},
             'ws3': {'f090': 13.801020973253475, 'f150': 10.621479083454684},
             'ws4': {'f090': 13.801020973253475, 'f150': 14.161972111272913},
             'ws5': {'f090': 11.291744432661936, 'f150': 9.996686196192643},
             'ws6': {'f090': 17.016658860021536, 'f150': 12.915718565480894},
         }
         }
        return calibration_factors[self.telescope.lower()]

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
        freq = next((match for match in matches if match.startswith('f')),
                    None)
        return ws, freq

    def bundle(self, bundle_id, split_label=None, null_prop_val=None,
               map_dir=None, abscal=False, nproc=1):
        """
        Make a map bundle given a bundle ID and, optionally, null properties.

        Parameters
        ----------
        bundle_id: int
            ID corresponding to the bundle that observations belong to.
        split_label: str
            String, e.g. "detector_left" or "fast_scan"
            indicating the intra-obs null split that observations belong to.
        null_prop_val: str
            String of format "{quality}_{null_property}", e.g. "low_pwv"
            indicating the inter-observation null split that observations
            belong to.
        abscal: bool
            Apply saved abscal factors if True.
        nproc: int
            Number of parallel processes to use. 1 for serial.

        Returns
        -------
        signal: np.array
            Output bundled signal map.
        weights: np.array
            Output bundled weights map.
        hits: np.array
            Output bundled hits map.
        """
        fnames = self._get_fnames(bundle_id, null_prop_val, split_label, map_dir=map_dir)
        # DEBUG
        if len(set(fnames)) < len(fnames):
            raise ValueError("fnames contains duplicates")
        print(
            f"{len(fnames)} atomic file names (bundle {bundle_id})"
        )

        abfac = np.array([self.get_abfac(fname, abscal) for fname in fnames]) if abscal == True else 1  # noqa
        weights_list = [fname.replace("wmap", "weights") for fname in fnames]
        hits_list = [fname.replace("wmap", "hits") for fname in fnames]

        signal, weights, hits = coadd_maps(
            fnames, weights_list, hits_list, pix_type=self.pix_type,
            car_template_map=self.car_map_template, abscal=abfac, nproc=nproc
            )

        return signal, weights, hits, fnames


class SignFlipper(_Coadder):
    """
    Child class of _Coadder, with the purpose of sign-flipping and coadding
    atomic maps for the purpose of generating per-bundle noise maps.
    """
    def __init__(self, atomic_db, bundle_db, freq_channel, car_map_template, 
                 split_label=None, wafer=None, abscal=False, #nproc=1
                 bundle_id=None, null_prop_val=None, pix_type="hp", map_dir=None):
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
                         pix_type=pix_type, car_map_template=car_map_template)
        
        self.fnames, self.ws = self._get_fnames(bundle_id, null_prop_val, split_label, return_weights=True, map_dir=map_dir)
        
        # TODO: temporary workaround to remove maps with unphysically low weights (flat maps)
        self.fnames, self.ws = zip(*[(fname, w) for fname, w in zip(self.fnames, self.ws) if w < 2e+10])
        
        # DEBUG
        if len(set(self.fnames)) < len(self.fnames):
            raise ValueError("fnames contains duplicates")
        print(
            f"{len(self.fnames)} atomic file names (bundle {bundle_id})"
        )

        abfac = np.array([self.get_abfac(fname, abscal) for fname in self.fnames]) if abscal == True else 1  # noqa

        self.wmaps = [read_map(fname, pix_type=self.pix_type,
                               fields_hp=self.fields_hp)
                      for fname in self.fnames]
        
        self.weights = [read_map(fname.replace("wmap", "weights"),
                                 pix_type=self.pix_type,
                                 fields_hp=self.fields_hp,
                                 is_weights=True)
                        for fname in self.fnames]
        

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

        signflip_noise, weights_noise = coadd_maps(
            maps_list, weights_list, pix_type=self.pix_type,
            sign_list=sign_list, car_template_map=self.car_map_template, 
            #nproc=nproc
        )

        return signflip_noise, weights_noise
