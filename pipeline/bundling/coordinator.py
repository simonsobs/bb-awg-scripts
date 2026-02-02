import numpy as np
import pandas as pd
import sqlite3
from bundling_utils import filter_by_atomic_list
import os

class BundleCoordinator:
    def __init__(self, atomic_db=None, n_bundles=None, seed=None,
                 null_props=None, query_restrict="", atomic_list=None,
                 weight='median_weight_qu'):
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
        null_props: dict
            Keys should be strings of (inter-obs null test) props available in
            atomic db.
            Values can be:
              - "median" to separate into two groups based on median values
              - {"splits": val_splits, "names": [name1, name2, ...]}
              - None to use each string value in the atomic db as its own group
            val_splits can be:
              - [(min1, max1), (min2, max2), ...] to pick vals in a numerical
                range
              - [(str1, str2, ...), (str3, str4, ...)] to group string values
        query_restrict: str
            SQL query to restrict obs from the atomic database
        atomic_list: list
            External list of string tuples (obs_id, wafer, freq_channel) of
            atomic maps that are to be used for the bundling.
            All other atomics in atomic_db will be left out.
        weight: str
            Column name in the atomic_db of the atomic map weight you want to use
       """
        if atomic_db is None:
            return
        self.atomic_db = atomic_db
        conn = sqlite3.connect(atomic_db)
        cursor = conn.cursor()
        db_props = cursor.execute(
            "SELECT name FROM PRAGMA_TABLE_INFO('atomic')"
        ).fetchall()
        cursor.close()
        db_props = [prop[0] for prop in db_props]
        # print("Available info atomic_maps.db: ", db_props)

        if query_restrict != "":
            query_restrict = f" WHERE split_label='science' AND ({query_restrict})"  # noqa
        else:
            query_restrict = " WHERE split_label='science'"

        # Get medians and expand null props with full split info
        _check_null_props(null_props, db_props)
        self.null_props = _update_null_props(null_props, conn, query_restrict=query_restrict, atomic_list=atomic_list)

        self.to_query = ["obs_id", "timestamp"] + [null_prop for null_prop in self.null_props]

        # Get obs_id, wafer, freq_channel, split_label for all valid atomics and filter through atomic_list
        query1 = "SELECT obs_id, wafer, freq_channel FROM atomic" + query_restrict
        valid_waferbands = pd.read_sql_query(query1, conn).to_numpy()  # Good wafer-bands from query_restrict
        valid_waferbands = filter_by_atomic_list(valid_waferbands, atomic_list)  # Filter wafer-bands through atomic list
        to_query2 = "obs_id, wafer, freq_channel, split_label, prefix_path"
        to_query2 += f", {weight}" if weight else ""
        query2 = f"SELECT {to_query2} FROM atomic WHERE valid=1"
        valid_splits = pd.read_sql_query(query2, conn)

        atomics = filter_by_atomic_list(valid_splits, valid_waferbands)
        if weight:
            atomics=atomics.rename(columns={weight: 'weight'})
        else:
            atomics['weight'] = np.full(atomics.shape[0], None)
        self.atomics = atomics

        # Get all props used to assign bundle properties
        query = f"SELECT {', '.join(self.to_query)} FROM atomic"
        query = query.replace("timestamp", "ctime")
        query += query_restrict
        cursor = conn.cursor()
        all_props = cursor.execute(query).fetchall()  # First load directly to deal with timestamp/ctime rename
        all_props = pd.DataFrame(all_props, columns=self.to_query)
        cursor.close()
        conn.close()

        unique_indices = np.unique(all_props.obs_id, return_index=True)[1]
        all_props = all_props.loc[unique_indices]
        all_props = filter_by_atomic_list(all_props, atomic_list, obs_id_only=True)

        # Replace numerical null_props data from atomic db with labels for bundle db
        self.bundle_db = pd.DataFrame()
        for prop in all_props:
            if self.null_props is None or prop not in self.null_props:
                self.bundle_db[prop] = all_props[prop]
            else:
                null_dict = self.null_props[prop]
                self.bundle_db[prop] = _get_null_labels(null_dict, all_props[prop])
            setattr(self, prop, self.bundle_db[prop].to_numpy())

        self.n_bundles = n_bundles
        self.seed = seed

        self.gen_bundles()
        self.bundle_db['bundle_id'] = self.bundle_id

        metadata = _get_metadata(self.null_props)
        self.metadata = pd.DataFrame.from_dict(metadata, orient='columns')

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

        db_props = cursor.execute(
            "SELECT name FROM PRAGMA_TABLE_INFO('bundles')"
        ).fetchall()
        cursor.close()
        db_props = [prop[0] for prop in db_props]

        query_fmt = ",".join(db_props)
        add_query = ""
        if bundle_id is not None:
            keyword = "WHERE" if add_query == "" else "AND"
            add_query += f" {keyword} bundle_id = {bundle_id}"
        if null_prop_val not in [None, "science"]:
            for npv in np.atleast_1d(null_prop_val):
                keyword = "WHERE" if add_query == "" else "AND"
                null_prop_name = "_".join(npv.split("_")[1:])
                if null_prop_name not in db_props:
                    raise ValueError(f"null_prop_name {null_prop_name} "
                                     "not in db_props")
                add_query += f" {keyword} {null_prop_name} = '{npv}'"
        query = f"SELECT {query_fmt} FROM bundles{add_query}"
        results = pd.read_sql_query(query, db_con)

        if 'timestamp' not in db_props and 'ctime' in db_props:  # Compatibility with old dbs
            results['timestamp'] = results['ctime']
        # Handle empty results
        if results.size == 0:
            raise ValueError("Query returned no results.")

        bundle_coord = cls()
        for prop in results:
            setattr(bundle_coord, prop, results[prop].to_numpy())
        bundle_coord.bundle_db = results
        bundle_coord.n_bundles = len(list(set(bundle_coord.bundle_id)))

        atomics = pd.read_sql_query("SELECT * FROM atomic", db_con)
        obs_id = np.unique(bundle_coord.obs_id)
        atomics = filter_by_atomic_list(atomics, obs_id, obs_id_only=True)
        bundle_coord.atomics = atomics

        bundle_coord.metadata = pd.read_sql_query("SELECT * FROM metadata", db_con)

        db_con.close()
        return bundle_coord

    def gen_bundles(self):
        """
        Assign bundle ids for each obs_id and store in self.bundle_id.
        bundle_ids are generated then randomly assigned to obs, enforcing
        equal size bundles.
        """
        gen = np.random.default_rng(seed=self.seed)
        bundle_id = np.arange(self.bundle_db.shape[0]) % self.n_bundles
        self.bundle_id = gen.permutation(bundle_id)
        return self.bundle_id

    def get_ctimes(self, bundle_id, null_prop_val=None):
        """
        Get all ctimes corresponding to the obs with a given bundle_id
        and optionally null_prop_val.
        """
        filter = (self.bundle_id == int(bundle_id))
        timestamps = self.timestamp[filter]
        if null_prop_val not in [None, "science"]:
            name_prop = "_".join(null_prop_val.split("_")[1:])
            prop_val = getattr(self, name_prop)[filter]
            filter = prop_val == null_prop_val
            timestamps = timestamps[filter]
        return timestamps

    def save_db(self, db_path, overwrite=True):
        """
        Save the db to a file.
        The db contains three tables:
          - bundles: obs_id, timestamp, [all null props], bundle_id
          - metadata: name, min, max, tags for all null splits (e.g. low_pwv, high_pwv, etc.)
          - atomics: obs_id, wafer, freq_channel, split_label, weight, prefix_path for all
            available atomic maps matching selection criteria.
        """
        if os.path.exists(db_path):
            if overwrite:
                os.remove(db_path)
            else:
                raise ValueError(f"Bundle db {db_path} exists and overwrite=False. Exiting.")

        db_con = sqlite3.connect(db_path)

        # Save main bundles table
        self.bundle_db.to_sql("bundles", db_con, index=False)

        # Make metadata table containing info about splits
        self.metadata.to_sql("metadata", db_con, index=False)

        # Save atomic table with info about available atomic maps
        self.atomics.to_sql("atomic", db_con, index=False)

        db_con.commit()
        db_con.close()

def _get_metadata(null_props):
    """
    Extract metadata about splits from config null_props.
    Returns a dict with entries 'name', 'min', 'max', 'tags'.
    Each is a list containing the relevant info for all null splits.
    """
    metadata = {'name': [], 'min': [], 'max': [], 'tags': []}
    for prop, null_dict in null_props.items():
        for isplit in range(len(null_dict['splits'])):
            split = null_dict['splits'][isplit]
            if np.issubdtype(type(split[0]), np.number):
                split_data = [split[0], split[1], None]
            elif np.issubdtype(type(split[0]), np.str_):
                split_data = [None, None, ", ".join(split)]
            else:
                raise TypeError(f"Unrecognized split type {type(split[0])}")

            metadata['name'].append(null_dict['names'][isplit])
            metadata['min'].append(split_data[0])
            metadata['max'].append(split_data[1])
            metadata['tags'].append(split_data[2])
    return metadata

def _get_null_labels(null_dict, col):
    """Helper function to assign split labels to a column of data"""
    out = np.full(col.shape, None)
    for isplit in range(len(null_dict['splits'])):
        split = null_dict['splits'][isplit]
        # We expect a tuple of numbers
        if len(split) == 2 and np.issubdtype(type(split[0]), np.number):  # noqa
            out[np.logical_and(split[0] <= col, col < split[1])] = null_dict['names'][isplit]
        # Or a tuple of strings
        elif np.issubdtype(type(split[0]), np.str_):
            out[np.isin(col, split)] = null_dict['names'][isplit]
    return out

def _check_null_props(null_props, db_props):
    """ Helper function to check all null_props are in the db"""
    if null_props is None:
        return
    for null_prop in null_props:
        if null_prop not in db_props:
            raise ValueError(f"Property {null_prop} "
                             "not found in the database.")

def _update_null_props(null_props, conn, query_restrict="", atomic_list=None):
    """Helper function to update null_props with full information from atomic db.
        Replaces 'median' entries with full {'splits': ..., 'names': ...} format.

        Parameters
        ----------
        null_props: dict
            See BundleCoordinator.__init__
        conn: sqlite3.Connection
            Connection to an atomic db
        query_restrict: str
            SQL query to restrict obs from the atomic database
        atomic_list: list
            See BundleCoordinator.__init__
    """

    null_props = {} if null_props is None else null_props
    out_null_props = null_props.copy()
    for null_prop, null_val in null_props.items():
        query = f"SELECT obs_id, {null_prop} FROM atomic" + query_restrict
        res = pd.read_sql_query(query, conn)
        res = filter_by_atomic_list(res, atomic_list, obs_id_only=True)
        res = getattr(res, null_prop).to_numpy()

        if np.all(res == None):  # noqa
            raise ValueError(
                f"All values for property {null_prop} are None."
            )
        if np.issubdtype(res.dtype, np.number):
            if null_val == "median":
                med = np.median(res)
                out_null_props[null_prop] = {
                    "splits": [(-np.inf, med), (med, np.inf)],
                    "names": [f"low_{null_prop}",
                              f"high_{null_prop}"]
                }
        elif np.issubdtype(res.dtype, np.str_):
            if null_val is None:
                # Do 1-1 string matching
                all_vals = np.unique(res).tolist()  # noqa
                out_null_props[null_prop] = {
                    "splits": [(x,) for x in all_vals],
                    "names": all_vals
                }
    return out_null_props
