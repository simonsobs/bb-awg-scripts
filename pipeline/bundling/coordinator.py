import numpy as np
import sqlite3


class BundleCoordinator:
    def __init__(self, atomic_db=None, n_bundles=None, seed=None,
                 null_props=None, bundle_id=None, query_restrict=""):
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
            List of (inter-observation null test) properties
            that will be used to separate the atomic maps into two groups
            based on their median values, e.g. ["pwv", ...]
        """
        if atomic_db is not None:
            self.atomic_db = sqlite3.connect(atomic_db)
            # Load obs_id list from the database
            cursor = self.atomic_db.cursor()
            db_props = cursor.execute(
                "SELECT name FROM PRAGMA_TABLE_INFO('atomic')"
            ).fetchall()
            db_props = [prop[0] for prop in db_props]
            print("Available info atomic_maps.db: ", db_props)
            self.to_query = {"obs_id": "INTEGER",
                             "ctime": "INTEGER"}

            self.bundle_id = bundle_id
            self.null_props_stats = None

            if query_restrict != "":
                query_restrict = f" WHERE split_label='science' AND ({query_restrict})"
            else:
                query_restrict = " WHERE split_label='science'"
            if null_props is not None:
                self.null_props_stats = {}
                for null_prop in null_props:
                    if null_prop in db_props:
                        self.to_query[null_prop] = "TEXT"
                    else:
                        raise ValueError(f"Property {null_prop} "
                                         "not found in the database.")
                    query = f"SELECT {null_prop} FROM atomic" + query_restrict
                    res = np.asarray(
                        cursor.execute(query).fetchall()
                    ).flatten()
                    if np.all(res == None):  # noqa
                        raise ValueError(
                            f"All values for property {null_prop} are None."
                        )
                    if np.issubdtype(res.dtype, np.number):
                        self.null_props_stats[null_prop] = np.median(res)
                    elif np.issubdtype(res.dtype, np.str_):
                        self.null_props_stats[null_prop] = np.unique(res).tolist()  # noqa

            query = f"SELECT {', '.join(self.to_query.keys())} FROM atomic"
            query += query_restrict
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

        db_props = cursor.execute(
            "SELECT name FROM PRAGMA_TABLE_INFO('bundles')"
        ).fetchall()
        db_props = [prop[0] for prop in db_props]

        query_fmt = ",".join(db_props)
        add_query = ""
        if bundle_id is not None:
            keyword = "WHERE" if add_query == "" else "AND"
            add_query += f" {keyword} bundle_id = {bundle_id}"
        if null_prop_val not in [None, "science"]:
            keyword = "WHERE" if add_query == "" else "AND"
            null_prop_name = "_".join(null_prop_val.split("_")[1:])
            add_query += f" {keyword} {null_prop_name} = '{null_prop_val}'"
        query = cursor.execute(f"SELECT {query_fmt} FROM bundles{add_query}")
        results = np.asarray(query.fetchall())

        db_con.close()

        bundle_coord = cls()

        # Handle empty results
        if results.size == 0:
            raise ValueError("Query returned no results.")

        for i, prop in enumerate(db_props):
            if prop == "bundle_id":
                setattr(bundle_coord, "bundle_ids", results[:, i].astype(int))
            else:
                setattr(bundle_coord, prop, results[:, i])

        # bundle_coord.to_query = db_props
        bundle_coord.relevant_props = results
        bundle_coord.n_bundles = len(list(set(bundle_coord.bundle_ids)))

        return bundle_coord

    def gen_bundles(self):
        """
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        shuffled_props = np.random.permutation(self.relevant_props)
        bundle_ids = np.repeat(np.arange(self.n_bundles),
                               len(shuffled_props) // self.n_bundles)
        if len(shuffled_props) % self.n_bundles != 0:
            bundle_ids = np.concatenate([
                bundle_ids,
                np.arange(len(shuffled_props) % self.n_bundles)
            ])
        for i, prop in enumerate(self.to_query):
            setattr(self, prop, shuffled_props[:, i])
        self.shuffled_props = shuffled_props
        self.bundle_ids = bundle_ids

    def get_ctimes(self, bundle_id, split_label, null_prop_val=None):
        """
        """
        filter = (self.bundle_ids == int(bundle_id))
        ctimes = self.ctime[filter]
        if null_prop_val not in [None, "science"]:
            print(f"WARNING: you selected atomics with {null_prop_val} "
                  f"but you also selected a split_label {split_label}. "
                  "/n - Are you sure you want to do this?")
            name_prop = null_prop_val.split("_")[1]
            prop_val = getattr(self, name_prop)[filter]
            filter = prop_val == null_prop_val
            ctimes = ctimes[filter]

        return ctimes

    def save_db(self, db_path, overwrite=True):
        """
        """
        if overwrite:
            import os
            try:
                os.remove(db_path)
            except OSError:
                pass
        else:
            raise NotImplementedError("Can only overwrite db_path for now.")

        db_con = sqlite3.connect(db_path)
        bundle_db = db_con.cursor()
        names_and_fmt = [f"{prop_name} {prop_type}"
                         for prop_name, prop_type in self.to_query.items()]
        table_format = ",".join(names_and_fmt)
        table_format += ",bundle_id INTEGER"

        bundle_db.execute(f"CREATE TABLE bundles ({table_format})")

        db_data = []
        for id_row, row in enumerate(self.shuffled_props):
            dbrow = []
            for id_prop, prop in enumerate(self.to_query):
                null_stats = self.null_props_stats
                if null_stats is None or prop not in null_stats:
                    dbrow.append(row[id_prop])
                else:
                    try:
                        val = np.float64(row[id_prop])
                    except:  # noqa
                        val = row[id_prop]
                    if np.issubdtype(type(val), np.number):
                        if val <= null_stats[prop]:
                            null_name = f"low_{prop}"
                        else:
                            null_name = f"high_{prop}"
                    else:
                        null_name = val
                    dbrow.append(null_name)

            dbrow.append(int(self.bundle_ids[id_row]))
            db_data.append(dbrow)

        table_format = ",".join(["?" for _ in range(len(self.to_query) + 1)])
        bundle_db.executemany(f"INSERT INTO bundles VALUES ({table_format})",
                              db_data)

        db_con.commit()
        db_con.close()
