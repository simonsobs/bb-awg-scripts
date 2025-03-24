import numpy as np
import sqlite3
import os
from bundling_utils import read_map, coadd_maps
from coordinator import BundleCoordinator
import re
from pixell import enmap

def map_union(map1, map2):
	"""Given two maps with compatible wcs but possibly covering different
	parts of the sky, return a new map that contains all pixels of both maps.
	If the input maps overlap, then those pixels will have the sum of the two maps"""
	oshape, owcs = enmap.union_geometry([map1.geometry, map2.geometry])
	omap = enmap.zeros(map1.shape[:-2]+oshape[-2:], owcs, map1.dtype)
	omap.insert(map1)
	omap.insert(map2, op=lambda a,b:a+b)
	return omap

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
            subquery = "ctime, wafer, freq_channel, prefix_path, median_weight_qu, ra_centre, dec_centre"
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
                    #f"{os.path.basename(prefix_path)}_wmap{suffix}"  # noqa
                    f"atomic_{str(ctime)}_{wafer}_{freq_channel}_{split_label}_wmap{suffix}"
                )
                for ctime, wafer, freq_channel, prefix_path, _, _, _ in result
            ]
            weights = [weight for _, _, _, _, weight, _, _ in result]
            ras = [ra for _, _, _, _, _, ra, _ in result]
            decs = [dec for _, _, _, _, _, _, dec in result]
            
            return fnames, weights, ras, decs
        else:
            fnames = [
                os.path.join(
                    map_dir, f"{str(ctime)[:5]}",
                    #f"{os.path.basename(prefix_path)}_wmap{suffix}"  # noqa
                    f"atomic_{str(ctime)}_{wafer}_{freq_channel}_{split_label}_wmap{suffix}"
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
            ra_list = []
            dec_list = []

        if (split_label is None) or isinstance(split_label, str):
            split_labels = [split_label]
        else:
            split_labels = split_label

        for obs_id in obs_ids:
            for split_label in split_labels:
                if return_weights:
                    fname_list, weight_list, ras, decs = self._obsid2fnames(
                        obs_id,
                        return_weights=return_weights,
                        split_label=split_label,
                        map_dir=map_dir
                    )
                    for fname, weight, ra, dec in zip(fname_list, weight_list, ras, decs):
                        if self._check_maps_exist(fname):
                            fnames.append(fname)
                            weights.append(weight)
                            ra_list.append(ra)
                            dec_list.append(dec)
                        else:
                            print(f"Missing map: {fname} !")
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
            return fnames, weights, ra_list, dec_list
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


class SignFlipper(Bundler):
    """
    Child class of Bundler, with the purpose of sign-flipping and coadding
    atomic maps for the purpose of generating per-bundle noise maps.
    """
    def load_maps(self, map_dir, bundle_id, split_labels, null_prop_val=None):
        """
        """
        self.wmaps, self.weights = [], []
        self.ras, self.decs = [], []

        for split_label in split_labels:

            fnames, ws, ras, decs = self._get_fnames(
                bundle_id,
                null_prop_val,
                return_weights=True,
                map_dir=map_dir,
                split_label=split_label
            )
            ras = np.rad2deg(ras)
            decs = np.rad2deg(decs)

            self.wmaps.append([
                read_map(
                    fname,
                    pix_type=self.pix_type,
                    fields_hp=self.fields_hp
                )
                for fname in fnames
            ])
            self.weights.append([
                read_map(
                    fname.replace("wmap", "weights"),
                    pix_type=self.pix_type,
                    fields_hp=self.fields_hp,
                    is_weights=True
                )
                for fname in fnames
            ])            
            self.ras.append(ras)
            self.decs.append(decs)

        if len(split_labels) == 2:
            
            wmaps, weights = [], []
            for i in range(len(self.wmaps[0])):
                wm1 = self.wmaps[0][i]
                wm2 = self.wmaps[1][i]
                w1 = self.weights[0][i]
                w2 = self.weights[1][i]
                
                wm = map_union(wm1, wm2)
                w = map_union(w1, w2)
    
                #good = w > 0
                #wm[good] /= w[good]
                wmaps.append(wm)
                weights.append(w)

            self.wmaps = wmaps
            self.weights = weights

            self.ras = np.array([(ra1 + ra2) / 2 for ra1, ra2 in zip(self.ras[0], self.ras[1])])
            self.decs = np.array([(dec1 + dec2) / 2 for dec1, dec2 in zip(self.decs[0], self.decs[1])])

        self.ws = [
            np.median(w[1:][w[1:] > 0]) for w in self.weights
        ]


    def batch_inputs(self, bins_ra, bins_dec):
        """
        """
        ra_perc = np.linspace(0, 100, bins_ra + 1)[1:-1]
        dec_perc = np.linspace(0, 100, bins_dec + 1)[1:-1]

        ra_lims = np.percentile(self.ras, ra_perc)
        ra_lims = np.concatenate(([self.ras.min()], ra_lims, [self.ras.max()]))
        
        dec_lims_list = []
        for i in range(bins_ra):
            dec_sub = self.decs[(self.ras >= ra_lims[i]) & (self.ras < ra_lims[i + 1])]
            dec_lims = np.percentile(dec_sub, dec_perc)
            dec_lims = np.concatenate(([dec_sub.min()], dec_lims, [dec_sub.max()]))
            dec_lims_list.append(dec_lims)

        # Batch maps
        wmaps = []
        weights = []
        ws = []
        for i in range(bins_ra):
            for j in range(bins_dec):
                if i == bins_ra - 1:
                    ra_mask = (self.ras >= ra_lims[i]) & (self.ras <= ra_lims[i + 1])
                else:
                    ra_mask = (self.ras >= ra_lims[i]) & (self.ras < ra_lims[i + 1])
                
                if j == bins_dec - 1:
                    dec_mask = (self.decs >= dec_lims_list[i][j]) & (self.decs <= dec_lims_list[i][j + 1])
                else:
                    dec_mask = (self.decs >= dec_lims_list[i][j]) & (self.decs < dec_lims_list[i][j + 1])
                tot_mask = ra_mask & dec_mask

                wmaps.append([self.wmaps[k] for k in range(len(tot_mask)) if tot_mask[k]])
                weights.append([self.weights[k] for k in range(len(tot_mask)) if tot_mask[k]])
                ws.append([self.ws[k] for k in range(len(tot_mask)) if tot_mask[k]])
        self.wmaps = wmaps
        self.weights = weights
        self.ws = ws


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

        wmaps = []
        weights = []
        signs = []
        for wmap_group, weights_group, w_group in zip(self.wmaps, self.weights, self.ws):
            print("GROUP LENGTH is ", len(wmap_group))
            perm_idx = np.random.permutation(len(wmap_group))
            maps_list, weights_list = ([wmap_group[i] for i in perm_idx],
                                          [weights_group[i] for i in perm_idx])
            mean_wQU_list = [w_group[i] for i in perm_idx]
            weight_cumsum = np.cumsum(mean_wQU_list) / np.sum(mean_wQU_list)
            pm_one = np.random.choice([-1, 1])
            sign_list = np.where(weight_cumsum < 0.5, pm_one, -pm_one)

            signs = np.concatenate((signs, sign_list))
            wmaps += maps_list
            weights += weights_list

        signflip_noise = coadd_maps(
            wmaps, weights, pix_type=self.pix_type,
            sign_list=signs, abscal=1.,
        )[0]

        return signflip_noise
