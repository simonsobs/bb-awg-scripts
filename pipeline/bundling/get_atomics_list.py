import argparse
import sqlite3
import numpy as np
import os


def get_atomic_maps_list(map_dir, queries_list, map_type="wmap", db_fpath=None,
                         verbose=False, ext="fits.gz"):
    """
    Outputs list of atomic maps
    (e.g. "{map_dir}/17107/atomic_1710705196_ws0_f090_full_{map_type}.fits")
    based on map_dir (base directory for atomic maps) and queries_list.

    Parameters
    ----------
    map_dir: str
        Path to directory of atomic maps and atomics database.
    db_fpath: str
        Full path to file name of atomic maps database for map_dir. If not
        provided, default to "{map_dir}/atomic_maps.db". Default: None
    map_type: str
        Map type to output, e.g. 'wmap' fo weighted map, 'weights' for inverse
        variance weights map, 'hits' for hits map. Default: 'wmap'.
    queries_list: list of str
        List with SQL query strings, e.g.
        "freq_channel = 'f090'", "elevation >= '50'", etc.
    verbose: boolean
        Verbose output? Default: False

    Returns
    -------
    atomic_maps_prefix_list: list of str
        List with path prefixes to all atomic maps requested.
    """
    if db_fpath is None:
        fpath = f"{map_dir}/atomic_maps.db"
        if os.path.isfile(fpath):
            db_fpath = fpath
    if verbose:
        if db_fpath:
            print(f" Querying {db_fpath}")
        else:
            print("WARNING: No atomics database found. "
                  "Ignoring ALL SQL queries.")
    assert map_type in ["wmap", "weights", "hits"]

    if not db_fpath:
        map_template = "atomic_*_{wafer_slot}_{freq_channel}_*_{map_type}.{ext}"  # noqa

        props = {}

        for query in queries_list:
            # TODO: other props?
            prop_name, prop_value = query.split(" = ")
            prop_value = prop_value.strip("'")
            props[prop_name] = prop_value

        for prop in ["wafer_slot", "freq_channel"]:
            if prop not in props:
                props[prop] = "*"

        map_template = map_template.format(map_type=map_type, ext=ext, **props)

        # Simply loop over all .fits.gz files with matching file names.
        import glob

        globbing = f"{map_dir}/**/{map_template.format(map_type=map_type)}"
        return glob.glob(globbing, recursive=True)

    # Otherwise, execute SQL query
    atomic_maps_list = []
    query_target = "prefix_path"
    table_name = "atomic"
    where_statement = " AND ".join(queries_list)

    conn = sqlite3.connect(db_fpath)
    cursor = conn.cursor()

    # Print columns
    data = cursor.execute(f"SELECT * FROM {table_name};")
    cols = [col[0] for col in data.description]
    if verbose:
        print(f"Columns of {table_name}:\n  ", " ".join(cols))

    # FIXME: Give warning if query key is not present in the database.

    # Read data
    cursor.execute(f"select {query_target} from {table_name} where {where_statement};")  # noqa
    result = cursor.fetchall()
    if verbose:
        print(f" Found {len(result)} matching entries.")

    for r in result:
        fname = f"{map_dir}/{'/'.join(r[0].split('/')[-2:])}_{map_type}.{ext}"
        if os.path.isfile(fname):
            atomic_maps_list.append(fname)
    conn.commit()
    conn.close()

    return atomic_maps_list


def main(args):
    """
    Get list of NERSC filepaths of atomic maps corresponding to an SQL query
    to the associated atomic maps database.
    """
    print(f"Reading atomics at {args.map_dir}")
    os.makedirs(args.outdir, exist_ok=True)
    atomic_maps_list = {}
    atomic_maps_list["wmap"] = get_atomic_maps_list(
        args.map_dir, args.queries_list, map_type="wmap", verbose=args.verbose, db_fpath=args.db_fpath, ext=args.ext
    )

    for typ in ["weights", "hits"]:
        atomic_maps_list[typ] = [a.replace("wmap", typ)
                                 for a in atomic_maps_list["wmap"]]
    natomics = len(atomic_maps_list["wmap"])
    np.savez(f"{args.outdir}/atomic_maps_list.npz", **atomic_maps_list)
    print(f"Stored list of {natomics} atomics "
          f"under {args.outdir}/atomic_maps_list.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_dir",
                        help="Atomic maps directory.")
    parser.add_argument("--queries_list", nargs="+", default=[],
                        help="String with SQL queries of form "
                        "\"freq_channel = 'f090'\".")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output?")
    parser.add_argument("--outdir",
                        help="Output directory for atomic maps list.")
    parser.add_argument("--db_fpath", default=None,
                        help="Path to atomic map db.")
    parser.add_argument("--ext", default="fits.gz",
                        help="File extension for the maps.")

    args = parser.parse_args()
    main(args)
