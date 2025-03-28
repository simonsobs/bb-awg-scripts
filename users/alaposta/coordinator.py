#!/bin/sh python3

import numpy as np
import healpy as hp
import h5py
import sqlite3


class BundleCoordinator:
    def __init__(self,
                 atomic_db=None,
                 n_bundles=None,
                 seed=None,
                 null_props=None):
        """
        Constructor for the BundleCoordinator class.
        If no atomic database path is provided, the
        class will only set the number of bundles and
        the seed. This is to allow for an alternative
        constructor for a BundleCoordinator object
        which creates it from an existing database.

        Parameters
        ----------
        atomic_db: str
            Path to the atomic database.
        n_bundles: int
            Number of bundles to be generated.
        seed: int
            Seed for the random number generator.
        null_props: list of str
            List of properties that will be used to
            separate the atomic maps into two groups
            based on their median values.
            e.g. ["pwv", ...]
        """
        if atomic_db is not None:
            self.atomic_db = sqlite3.connect(atomic_db)
            # Load obs_id list from the database
            cursor = self.atomic_db.cursor()
            db_props = cursor.execute("SELECT name FROM PRAGMA_TABLE_INFO('atomic')").fetchall()
            db_props = [prop[0] for prop in db_props]

            self.to_query = {"obs_id": "INTEGER",
                             "ctime": "INTEGER"}

            if null_props is not None:
                self.null_props_stats = {}
                for null_prop in null_props:
                    if null_prop in db_props:
                        self.to_query[null_prop] = "TEXT"
                    else:
                        raise ValueError(f"Property {null_prop} not found in the database.")
                    query = f"SELECT {null_prop} FROM atomic"
                    res = np.asarray(cursor.execute(query).fetchall()).flatten()
                    self.null_props_stats[null_prop] = np.median(res)

            query = f"SELECT {', '.join(self.to_query.keys())} FROM atomic"
            res = np.asarray(cursor.execute(query).fetchall())

            unique_indices = np.unique(res[:, 0], return_index=True)[1]
            res = res[unique_indices]

            self.relevant_props = res

            self.n_bundles = n_bundles
            self.seed = seed

            self.gen_bundles()

    @classmethod
    def from_dbfile(cls, db_path, bundle_id=None, null_prop_val=None):
        """
        Alternative constructor for the BundleCoordinator class.
        Creates a BundleCoordinator object from an existing
        database file.

        Parameters
        ----------
        db_path: str
            Path to the database file.
        
        Returns
        -------
        bundle_coord: BundleCoordinator
            BundleCoordinator object created from the database.
        """
        db_con = sqlite3.connect(db_path)
        cursor = db_con.cursor()

        db_props = cursor.execute("SELECT name FROM PRAGMA_TABLE_INFO('bundles')").fetchall()
        db_props = [prop[0] for prop in db_props]

        query_fmt = ",".join(db_props)
        add_query = ""
        if bundle_id is not None:
            keyword = "WHERE" if add_query == "" else "AND"
            add_query += f" {keyword} bundle_id = {bundle_id}"
        if null_prop_val is not None:
            keyword = "WHERE" if add_query == "" else "AND"
            null_prop_name = null_prop_val.split("_")[1]
            add_query += f" {keyword} {null_prop_name} = '{null_prop_val}'"
        query = cursor.execute(f"SELECT {query_fmt} FROM bundles{add_query}")
        results = np.asarray(query.fetchall())

        db_con.close()

        bundle_coord = cls()
        for i, prop in enumerate(db_props):
            setattr(bundle_coord, prop, results[:, i])

        return bundle_coord

    def gen_bundles(self):
        """
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        shuffled_props = np.random.permutation(self.relevant_props)
        bundle_ids = np.repeat(np.arange(self.n_bundles), len(shuffled_props) // self.n_bundles)
        if len(shuffled_props) % self.n_bundles != 0:
            bundle_ids = np.concatenate([
                bundle_ids,
                #np.random.choice(np.arange(self.n_bundles), len(shuffled_props) % self.n_bundles)
                np.arange(len(shuffled_props) % self.n_bundles)
            ])
        for i, prop in enumerate(self.to_query):
            setattr(self, prop, shuffled_props[:, i])
        self.shuffled_props = shuffled_props
        self.bundle_ids = bundle_ids

    def get_ctimes(self, bundle_id, null_prop_val=None):
        """
        """
        filter = (self.bundle_id == bundle_id)
        ctimes = self.ctime[filter]
        if null_prop_val is not None:
            name_prop = null_prop_val.split("_")[1]
            prop_val = getattr(self, name_prop)[filter]
            filter = prop_val == null_prop_val
            ctimes = ctimes[filter]

        return ctimes

    def save_db(self, db_path):
        """
        """
        db_con = sqlite3.connect(db_path)
        bundle_db = db_con.cursor()
        names_and_fmt = [f"{prop_name} {prop_type}" for prop_name, prop_type in self.to_query.items()]
        table_format = ",".join(names_and_fmt)
        table_format += ",bundle_id INTEGER"

        bundle_db.execute(f"CREATE TABLE bundles ({table_format})")

        db_data = []
        for id_row, row in enumerate(self.shuffled_props):
            dbrow = []
            for id_prop, prop in enumerate(self.to_query):
                if prop in self.null_props_stats:
                    null_name = f"low_{prop}" if np.float64(row[id_prop]) <= self.null_props_stats[prop] else f"high_{prop}"
                    dbrow.append(null_name)
                else:
                    dbrow.append(row[id_prop])
            dbrow.append(int(self.bundle_ids[id_row]))
            db_data.append(dbrow)

        table_format = ",".join(["?" for _ in range(len(self.to_query) + 1)])
        bundle_db.executemany(f"INSERT INTO bundles VALUES ({table_format})", db_data)

        db_con.commit()
        db_con.close()


# Following functions are utility function inherited
# from sat-mapbundle-lib but not yet used. Might be
# relevant when coadding the atomics.
def read_hdf5_map(fname, to_nest=False):
    """
    """
    f = h5py.File(fname, "r")
    dset = f["map"]
    header = dict(dset.attrs)

    if header["ORDERING"] == "NESTED":
        file_nested = True
    elif header["ORDERING"] == "RING":
        file_nested = False

    _, npix = dset.shape

    if file_nested and not to_nest:
        mapdata = hp.reorder(dset[:], n2r=True)
    elif not file_nested and to_nest:
        mapdata = hp.reorder(dset[:], r2n=True)
    else:
        mapdata = dset

    return mapdata


def write_hdf5_map(fname, nside, dict_maps, list_of_obsid,
                   nest_or_ring='RING'):
    """
    """
    with h5py.File(fname, 'w') as f:
        f.attrs['NSIDE'] = nside
        f.attrs['ORDERING'] = nest_or_ring
        f.attrs['OBS_ID'] = list_of_obsid
        for k, v in dict_maps.items():
            f.create_dataset(k, data=v)


def gen_masks_of_given_atomic_map_list_for_bundles(nmaps, nbundles):
    """
    Makes a list (length nbundles) of boolean lists (length nmaps)
    corresponding to the atomic maps that have to be coadded to make up a
    given bundle. This is done by uniformly distributing atomic maps into each
    bundle and, if necessary, looping through the bundles until the remainders
    have gone.

    Parameters
    ----------
    nmaps: int
        Number of atomic maps to distribute.
    nbundles: int
        Number of map bundles to be generated.

    Returns
    -------
    boolean_mask_list: list of list of str
        List of lists of booleans indicating the atomics to be coadded for
        each bundle.
    """

    n_per_bundle = nmaps // nbundles
    nremainder = nmaps % nbundles
    boolean_mask_list = []

    for idx in range(nbundles):
        if idx < nremainder:
            _n_per_bundle = n_per_bundle + 1
        else:
            _n_per_bundle = n_per_bundle

        i_begin = idx * _n_per_bundle
        i_end = (idx+1) * _n_per_bundle

        _m = np.zeros(nmaps, dtype=np.bool_)
        _m[i_begin:i_end] = True

        boolean_mask_list.append(_m)

    return boolean_mask_list
