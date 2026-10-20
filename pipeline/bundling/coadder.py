import numpy as np
import os
import bundling_utils as utils
from coordinator import BundleCoordinator
import re
import sqlite3
import pandas as pd

class _Coadder:
    def __init__(self, bundle_db, freq_channel, wafer=None,
                 pix_type="hp", atomic_list=None, car_map_template=None):
        """
        Constructor for the _Coadder class. Reads in map
        and bundling information from bundle_db.

        Parameters
        ----------
        bundle_db: str
            Path to the bundling database.
        freq_channel: str
            Frequency channel label indicating the frequency to be read from.
            For SAT1 and SAT3, possible choices are "f090", "f150".
        wafer: str
            Optional; label indicating the telescope wafer to include.
            If no wafer is provided, coadd maps made for all the wafers.
        pix_type : str
            Pixelization type. Admissible values are "hp", "car".
        atomic_list: list
            External list of string tuples (obs_id, wafer, freq_channel) of
            atomic maps that are to be used for the bundling.
        car_map_template: str
            Path to CAR map or geometry to be used as template
        """
        assert pix_type in ["hp", "car"]
        self.pix_type = pix_type
        self.fields_hp = range(3) if pix_type == "hp" else None

        self.bundle_db = bundle_db
        self.freq_channel = freq_channel
        self.wafer = wafer
        self.atomic_list = atomic_list
        self.car_map_template = car_map_template

        self.bundle_info = None
        self.info_meta = None

        if not isinstance(freq_channel, str):
            raise TypeError(f"Coadder arg freq_channel should be a string. It is {type(freq_channel)}: {freq_channel}")

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

    def _prefixpath2fname(self, path, map_dir=None, depth=1):
        """
        Replace prefix path with the given map_dir.

        Parameters
        ----------
        path: str
            prefix_path from atomic db: full map path except for final '_wmap.fits'
        map_dir: str
            New path to map directory
        depth: int
            How many levels of prefix_path to keep
            depth=0 keeps just the filename. depth=1 keeps one parent directory, etc.

        Returns
        -------
        new_path: str
            %s-style format string. Full file path will be new_path % maptype
            where maptype = 'wmap', 'weights', or 'hits'
        """
        # For now, assume HEALPix maps are gzipped fits and CAR maps are fits.
        suffix = ".fits.gz" if self.pix_type == "hp" else ".fits"
        if map_dir is None:
            # Keep the default path
            return f"{path}_%s{suffix}"
        else:
            split_prefix = [dr for dr in path.split('/') if dr]  # Split and remove double slashes
            basename = '/'.join(split_prefix[-depth-1:])
            return os.path.join(map_dir, basename) + f"_%s{suffix}"

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

    def _get_bundle_info(self,  bundle_id, map_dir, null_prop_val=None, split_label=None, cache=True):
        """
        Extract info from the bundle db and cache in self.bundle_info.
        info is all from atomics table:  obs_id, wafer, freq_channel,
        split_label, weight, filename.

        Parameters
        ----------
        bundle_id: int
            ID corresponding to the bundle that observations belong to.
        map_dir: str
            Path to directory containing atomic maps
        null_prop_val: str
            String of format "{quality}_{null_property}", e.g. "low_pwv"
            indicating the null split that observations belong to.
        split_label: str
            String, e.g. "detector_left" or "fast_scan"
            indicating the intra-obs null split that observations belong to.

        Returns
        -------
        info: pandas.DataFrame
             columns: obs_id, wafer, freq_channel, split_label, weight, filename.
        """

        info_meta = (bundle_id, null_prop_val, split_label)
        if (self.bundle_info is not None) and self.info_meta == info_meta:
            return self.bundle_info  # Load cached

        bundle_coord = BundleCoordinator.from_dbfile(
            self.bundle_db,
            bundle_id=bundle_id,
            null_prop_val=null_prop_val
        )
        info = bundle_coord.atomics

        # Add map dir
        info['filename'] = [self._prefixpath2fname(ppath, map_dir, depth=1) for ppath in info['prefix_path']]

        # Filter by split labels
        if split_label is not None:
            info = info[np.isin(info['split_label'], np.atleast_1d(split_label))]

        # Filter info by freq, wafer, atomic list
        info = info[info.freq_channel == self.freq_channel]
        if self.wafer is not None:
            info = info[info.wafer == self.wafer]
        if self.atomic_list is not None:
            info = utils.filter_by_atomic_list(info, self.atomic_list)

        if cache:
            self.bundle_info = info  # Cache
            self.info_meta = info_meta
        return info


    def _get_fnames(self, bundle_id, map_dir, null_prop_val=None, split_label=None,
                    return_weights=False):
        """
        Return file names (and, optionally, mean polarization weights) given a
        bundle_id and a null property. For parameters, see Coadder_get_bundle_info()
        """
        info = self._get_bundle_info(bundle_id, map_dir, null_prop_val=null_prop_val, split_label=split_label)
        if not return_weights:
            return info['filename'].to_numpy()
        else:
            return info['filename'].to_numpy(), info['weight'].to_numpy()


class Bundler(_Coadder):
    """
    Child class of _Coadder, with the purpose of coadding atomic maps for the
    purpose of generating map bundles and bundled hits maps.
    """

    def bundle(self, bundle_id, map_dir, split_label=None, null_prop_val=None,
               abscal=None, parallelizor=None):
        """
        Make a map bundle given a bundle ID and, optionally, null properties.

        Parameters
        ----------
        bundle_id: int
            ID corresponding to the bundle that observations belong to.
        map_dir: str
            Path to directory containing atomic maps
        split_label: str
            String, e.g. "detector_left" or "fast_scan"
            indicating the intra-obs null split that observations belong to.
        null_prop_val: str
            String of format "{quality}_{null_property}", e.g. "low_pwv"
            indicating the inter-observation null split that observations
            belong to.
        abscal: dict
            Nested dict in format {'f090': {'ws0': 1, 'ws1': 1,...}, ...}.
            Maps will be multiplied by the abscal factor.
        parallelizor: tuple
            (MPICommExecutor or ProcessPoolExecutor, as_completed_callable, num_workers)

        Returns
        -------
        signal: np.array
            Output bundled signal map.
        weights: np.array
            Output bundled weights map.
        hits: np.array
            Output bundled hits map.
        """
        bundle_info = self._get_bundle_info(bundle_id, map_dir, null_prop_val=null_prop_val, split_label=split_label)
        fnames = list(bundle_info['filename'])
        maps_list = [fname % "wmap" for fname in fnames]
        weights_list = [fname % "weights" for fname in fnames]
        hits_list = [fname % "hits" for fname in fnames]
        full_abscal = utils.get_abscal(abscal, bundle_info['wafer'], bundle_info['freq_channel'])
        signal, weights, hits = utils.coadd_maps(
            maps_list, weights_list, hits_list, pix_type=self.pix_type,
            car_template_map=self.car_map_template, abscal=full_abscal, parallelizor=parallelizor
            )

        return signal, weights, hits, fnames


class SignFlipper(_Coadder):
    """
    Child class of _Coadder, with the purpose of sign-flipping and coadding
    atomic maps for the purpose of generating per-bundle noise maps.
    """
    def __init__(self, bundle_db, freq_channel, car_map_template, map_dir,
                 split_label=None, wafer=None, abscal=None,
                 bundle_id=None, null_prop_val=None, pix_type="hp", atomic_list=None, method="obs_weight"):
        """
        Constructor for the SignFlipper class. Creates a SignFlipper object,
        given map and bundling information from bundle_db.

        Parameters
        ----------
        bundle_db: str
            Path to the bundling database.
        freq_channel: str
            Frequency channel label indicating the frequency to be read from.
            For SAT1 and SAT3, possible choices are "f090", "f150".
        car_map_template: str
            Path to CAR map or geometry to be used as template
        map_dir: str
            Path to directory containing atomic maps
        split_label: str
            String, e.g. "detector_left" or "fast_scan"
            indicating the intra-obs null split that observations belong to.
        wafer: str
            Optional; label indicating the telescope wafer to include.
            If no wafer is provided, coadd maps made for all the wafers.
        abscal: dict
            Nested dict in format {'f090': {'ws0': 1, 'ws1': 1,...}, ...}.
            Maps will be multiplied by the abscal factor.
        bundle_id: int
            Optional; ID corresponding to the bundle that observations belong
            to. If None, coadd maps from all bundles.
        null_prop_val: str
            String of format "{quality}_{null_property}", e.g. "low_pwv"
            indicating the null split that observations belong to.
        pix_type : str
            Pixelization type. Admissible values are "hp", "car.
        atomic_list: list
            External list of string tuples (obs_id, wafer, freq_channel) of
            atomic maps that are to be used for the bundling.
        method: str
            - 'obs_weight': Signs matched at obs level using the 'obs_weight' from bundle_db.
              Atomics from the same obs for all freqs/wafers/splits will have the same sign.
              Weight balancing attempted using the sum of science weights for all wafer/freqs.
            - 'split_weight': Within a given signflip (split/freq) all atomic with the same obs
              will have the same sign. Weight balancing with the combined weight of all such atomics.
            - 'random_obs': (obs_id, seed) is randomly mapped to a sign. No weight balancing.
        """
        super().__init__(bundle_db, freq_channel, wafer,
                         pix_type=pix_type, car_map_template=car_map_template, atomic_list=atomic_list)

        self.method = method
        if method not in ['obs_weight', 'split_weight', 'random_obs']:
            raise ValueError(f"Method {method} not recognized.")

        bundle_info = self._get_bundle_info(bundle_id, map_dir, null_prop_val=null_prop_val, split_label=split_label)
        self.fnames = list(bundle_info['filename'])
        # copy(): pandas >= 3.0 returns a read-only view, but ws is scaled in place below
        self.ws = bundle_info['weight'].to_numpy().copy()

        self.full_abscal = utils.get_abscal(abscal, bundle_info['wafer'], bundle_info['freq_channel'])
        self.ws = self.ws * (self.full_abscal**-2)  # ivar gets -2 powers of abscal
        self.wmaps = [utils.read_map(fname%'wmap',
                                     pix_type=self.pix_type,
                                     fields_hp=self.fields_hp)
                      for fname in self.fnames]

        print("wmaps: ", len(self.wmaps))

        self.weights = [utils.read_map(fname%'weights',
                                       pix_type=self.pix_type,
                                       fields_hp=self.fields_hp,
                                       is_weights=True)
                        for fname in self.fnames]
        print("weights: ", len(self.wmaps))
        self._weights_noise = None

        # Generate weights needed for obs-level balancing
        self.obs_ids = self.bundle_info.obs_id.to_numpy()
        self.unique_obs = np.unique(self.obs_ids)
        self.obs_inds = {obs_id: np.where(self.obs_ids == obs_id) for obs_id in self.unique_obs}
        if method == "obs_weight":
            conn = sqlite3.connect(self.bundle_db)
            bundle_id_query = f" WHERE bundle_id={bundle_id}" if bundle_id is not None else ""
            df = pd.read_sql_query("SELECT obs_id, obs_weight FROM bundles" + bundle_id_query + " ORDER BY obs_id", conn)
            conn.close()
            self.obs_weights = df['obs_weight'].to_numpy()
            self.obs_weight_ids = df['obs_id'].to_numpy()

        elif method == "split_weight":
            self.obs_weights = np.array([np.sum(bundle_info['weight'].to_numpy()[self.obs_inds[obs_id]]) for obs_id in self.unique_obs])
            self.obs_weight_ids = self.unique_obs

    def get_signs_from_weights(self, seed):
        weights = self.obs_weights
        gen = np.random.default_rng(seed)
        pm_one = gen.choice([-1, 1])
        perm_idx = gen.permutation(len(weights))
        weights_random = weights[perm_idx]
        weight_cumsum = np.cumsum(weights_random)/np.sum(weights_random)
        sign_list_obs = np.where(weight_cumsum < 0.5, pm_one, -pm_one)
        sign_list_obs = sign_list_obs[np.argsort(perm_idx)]  # Undo the permutation
        sign_list = np.zeros_like(self.obs_ids, dtype=int)
        for obs_id in self.unique_obs:
            idx = np.where(obs_id == self.obs_weight_ids)[0][0]
            sign_list[self.obs_inds[obs_id]] = sign_list_obs[idx]
        return sign_list

    def get_signs_from_obs(self, seed):
        if seed is None:
            raise ValueError("random_obs requires a seed to be passed")
        signs = []
        for obs_id in self.obs_ids:
            gen = np.random.default_rng(get_seed(obs_id, seed))
            signs.append(gen.choice([-1, 1]))
        return np.array(signs, dtype=int)


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
        class _NoInplaceScale:
            """Wrapper that prevents in-place scaling of cached maps in utils.sum_maps."""
            def __init__(self, base):
                self._base = base

            def __getattr__(self, name):
                return getattr(self._base, name)

            def __imul__(self, other):
                out = self._base.copy()
                out *= other
                return out

        if self.method in ['obs_weight', 'split_weight']:
            sign_list = self.get_signs_from_weights(seed)
        elif self.method  == 'random_obs':
            sign_list = self.get_signs_from_obs(seed)

        maps_list = [_NoInplaceScale(imap) for imap in self.wmaps]
        weights_list = [_NoInplaceScale(imap) for imap in self.weights]
        if self._weights_noise is None:
            signflip_noise, weights_noise = utils.coadd_maps(maps_list, weights_list,
                                                             pix_type=self.pix_type, sign_list=sign_list,
                                                             car_template_map=self.car_map_template,
                                                             abscal=self.full_abscal)
            self._weights_noise = weights_noise
            return signflip_noise, weights_noise

        signflip_noise = utils.sum_maps(
            maps_list,
            template=self._weights_noise,
            pix_type=self.pix_type,
            mult=sign_list * self.full_abscal**-1,
            fields_hp=self.fields_hp,
            )

        good_weights = self._weights_noise > 0
        signflip_noise[good_weights] /= self._weights_noise[good_weights]
        return signflip_noise, self._weights_noise

def get_seed(obs_id, seed0, MAXSEED=1e6):
    ctime = int(re.findall(r"(?<=obs_)\d+", obs_id)[0])
    if seed0 >= MAXSEED:
        raise ValueError("seed too large")
    return int(ctime * MAXSEED + seed0)
