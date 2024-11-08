# Playground for bundling code coadding
# utility classes. WIP
# Last updated: 24-10-30

import time
import numpy as np
import sqlite3
from coordinator import BundleCoordinator
import os
import healpy as hp

def _dbquery(db, query):
    cursor = db.cursor()
    result = cursor.execute(query).fetchall()
    return np.asarray(result).flatten()


class _Coadder:
    def __init__(self, atomic_db, bundle_db, freq_channel, wafer=None):
        """
        """
        self.atomic_db = atomic_db
        self.bundle_db = bundle_db
        self.freq_channel = freq_channel
        self.wafer = wafer

    def _get_obs_ids(self, bundle_id, null_prop_val=None):
        """
        """
        bundle_coord = BundleCoordinator.from_dbfile(
            self.bundle_db,
            bundle_id=bundle_id,
            null_prop_val=null_prop_val
        )
        obs_ids = bundle_coord.obs_id
        return obs_ids

    def _obsid2fnames(self, obs_id, return_weights=False):
        """
        """
        map_dir = os.path.dirname(self.atomic_db)

        con = sqlite3.connect(self.atomic_db)
        cursor = con.cursor()
        
        if return_weights:
            subquery = "ctime, wafer, freq_channel, median_weight_qu"
        else:
            subquery = "ctime, wafer, freq_channel"

        query = f"SELECT {subquery} FROM atomic WHERE freq_channel = '{self.freq_channel}' AND obs_id = '{obs_id}'"
        if self.wafer is not None:
            query += f" AND wafer = '{self.wafer}'"
        result = cursor.execute(query).fetchall()
        con.close()
        if return_weights:
            fnames = [
                os.path.join(map_dir, f"{str(ctime)[:5]}", f"atomic_{ctime}_{wafer}_{freq_channel}_full_wmap.fits.gz")
                for ctime, wafer, freq_channel, _ in result
            ]
            weights = [weight for _, _, _, weight in result]
            return fnames, weights
        else:
            fnames = [
                os.path.join(map_dir, f"{str(ctime)[:5]}", f"atomic_{ctime}_{wafer}_{freq_channel}_full_wmap.fits.gz")
                for ctime, wafer, freq_channel in result
            ]
            return fnames

    def _get_fnames(self, bundle_id, null_prop_val=None, return_weights=False):
        """
        """
        obs_ids = self._get_obs_ids(bundle_id, null_prop_val)
        fnames = []
        if return_weights:
            weights = []

        for obs_id in obs_ids:
            if return_weights:
                fname, weight = self._obsid2fnames(obs_id, return_weights=return_weights)
                fnames.extend(fname)
                weights.extend(weight)
            else:
                fname = self._obsid2fnames(obs_id, return_weights=return_weights)
                fnames.extend(fname)
        
        if return_weights:
            return fnames, weights
        return fnames


class Bundler(_Coadder):
    """
    """
    def bundle(self, bundle_id, null_prop_val=None):
        """
        """
        fnames = self._get_fnames(bundle_id, null_prop_val)

        for i, fname in enumerate(fnames):
            atom = hp.read_map(fname, field=[0, 1, 2])
            weights = hp.read_map(fname.replace("wmap", "weights"), field=[0, 1, 2])
            hits = hp.read_map(fname.replace("wmap", "hits"))

            if i == 0:
                layout = np.zeros_like(atom)
                layout_w = np.zeros_like(weights)
                layout_h = np.zeros_like(hits)
            layout += atom
            layout_w += weights
            layout_h += hits
        
        layout_w[layout_w == 0] = np.inf
        layout /= layout_w

        return layout, layout_h

class SignFlip(_Coadder):
    """
    """
    def __init__(self, atomic_db, bundle_db, freq_channel, wafer=None, bundle_id=None, null_prop_val=None):
        """
        """
        super().__init__(atomic_db, bundle_db, freq_channel, wafer)

        self.fnames, self.weights = self._get_fnames(bundle_id, null_prop_val, return_weights=True)
        self.wmaps = [hp.read_map(fname, field=[0, 1, 2]) for fname in self.fnames]
        self.ws = [hp.read_map(fname.replace("wmap", "weights"), field=[0, 1, 2]) for fname in self.fnames]


    def signflip(self, seed=None):
        """
        """
        if seed is not None:
            np.random.seed(seed)
        #fnames, weights = self._get_fnames(bundle_id, null_prop_val, return_weights=True)
        perm_idx = np.random.permutation(len(self.wmaps))
        maps, weights = np.array(self.wmaps)[perm_idx], np.array(self.ws)[perm_idx]

        weight_cumsum = np.cumsum(weights) / np.sum(weights)
        sign_list = np.where(weight_cumsum < 0.5, -1, 1)

        for i, (atom, weights) in enumerate(zip(self.wmaps, self.ws)):
            #atom = hp.read_map(fname, field=[0, 1, 2])
            #weights = hp.read_map(fname.replace("wmap", "weights"), field=[0, 1, 2])

            if i == 0:
                layout = np.zeros_like(atom)
                layout_w = np.zeros_like(weights)
            layout += atom * sign_list[i]
            layout_w += weights 
        
        layout_w[layout_w == 0] = np.inf
        layout /= layout_w

        return layout





b="""
class SignFlip:
    def __init__(self, state=None):
        self.prng = np.random.RandomState(int(1e+6*time.time()) % 2**32)
        if state:
            self.prng.set_state(state)
        self.state = self.prng.get_state()
        self.seq = None

    def gen_seq(self, obs_weights):
        self.state = self.prng.get_state()

        nums = len(obs_weights)
        obs = range(nums)
        obs_perm = self.prng.permutation(obs)

        obs_weights_perm = np.zeros_like(obs_weights)
        self.seq = np.zeros_like(obs_weights, dtype=np.bool_)

        for ob in obs_perm:
            obs_weights_perm[ob] = obs_weights[ob]

        w = obs_weights_perm
        wi = np.cumsum(w)
        wi = wi/np.max(wi)

        # no sign flip for the first half
        noflip = np.where(wi < 0.5)[0].tolist()
        # decide whether to flip the middle bundles by coin toss
        if len(w) > 1 and len(noflip) < 2 and self.prng.randint(0, 2):
            noflip.append(max(noflip)+1)

        for i in range(nums):
            if i in noflip:
                seq = False
            else:
                seq = True

            self.seq[obs_perm[i]] = seq

        return self.prng.get_state()"""
