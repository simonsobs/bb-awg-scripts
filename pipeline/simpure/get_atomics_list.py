import argparse
import os
import numpy as np

import qpoint as qp
import sqlite3
import matplotlib.pyplot as plt


def get_radec(az, el, ctime):
    """
    Get RA and Dec from azel with qpoint (function by E. Rosenberg)
    """
    Q = qp.QPoint()
    pitch = None
    roll = None
    # ~SO
    lat = -(22+57/60) * np.ones_like(ctime)
    lon = -(67+47/60) * np.ones_like(ctime)

    # calculate boresight quaternions
    q_bore = Q.azel2bore(az, el, pitch, roll, lon, lat, ctime)
    q_off = Q.det_offset(0, 0, 0)

    # calculate detector pointing
    ra, dec, _, _ = Q.bore2radec(q_off, ctime, q_bore)
    return ra, dec


def sqlite_db(path):
    if not os.path.isfile(path):
        print(f"{path} doesn't exist; creating new file.")

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # test if this is really sqlite file
    cur = conn.cursor()
    cur.execute('SELECT 1 from sqlite_master where type = "table"')
    try:
        cur.fetchone()
    except sqlite3.DatabaseError:
        msg = '%s can\'t be read as SQLite DB' % path
        raise argparse.ArgumentTypeError(msg)

    return conn


def copy_db_structure(src_db, dst_db):
    src_cur = src_db.cursor()
    dst_cur = dst_db.cursor()

    src_cur.execute('SELECT * from sqlite_master')
    src_master = src_cur.fetchall()

    src_tables = list(filter(lambda r: r['type'] == 'table', src_master))
    src_indices = list(filter(lambda r: r['type'] == 'index'
                              and r['sql'] is not None, src_master))

    for table in src_tables:
        print('Processing table:', table['name'])
        print('Delete old table in destination db, if exists')
        dst_cur.execute("DROP TABLE IF EXISTS " + table['name'])
        print('Creating table structure')
        dst_cur.execute(table['sql'])
        table_idx = list(filter(lambda r: r['tbl_name'] == table['name'],
                                src_indices))
        for idx in table_idx:
            dst_cur.execute(idx['sql'])

    src_db.close()
    dst_db.close()


def make_minimal_patch(atomics_dict, outdir, delta_ra, delta_dec,
                       ra_min=None, ra_max=None, dec_min=None, dec_max=None):
    """
    Loops over all atomics, then removes all atomics with (ra, dec) within
    a square of (delta_ra, delta_dec) within any other map in the list.

    Arguments
    ---------
        atomics_dict: dict
            items are lists of tuples containing RA, DEC, ctime sorted by RA
        outdir: str
            Output directory
        delta_ra: float
            minimum RA distance between two nearest observations
        delta_dec: float
            minimum DEC distance between two nearest observations
        ra_min: float
            Minimum right ascension of any observation
        ra_max: float
            Maximum right ascension of any observation
        dec_min: float
            Minimum declination of any observation
        dec_max: float
            Maximum declination of any observation

    Returns
    -------
        atomics_keep: dict
            items are lists of tuples containing RA, DEC, ctime sorted by RA
            corresponding to the kept entries
    """
    #wafer_list = [f"ws{i}" for i in range(7)]
    wafer_list = ["ws0"]

    colors = [f"C{i}" for i in range(7)]
    atomics_keep = {w: [] for w in wafer_list}

    for w in wafer_list:
        atomics = atomics_dict[w]

        # Start at minimum RA
        cur = (atomics[0][0], atomics[0][1])

        # Loop through ascending RA
        for ra, dec, ctime in atomics[1:]:
            next = (ra, dec)
            dra, ddec = (np.fabs(next[0] - cur[0]), np.fabs(next[1] - cur[1]))
            too_close = (dra < delta_ra and ddec < delta_dec)

            if None in [ra_min, ra_max, dec_min, dec_max]:
                outside_box = False
            else:
                outside_box = ((next[0] > ra_max) or (next[0] < ra_min)
                               or (next[1] > dec_max) or (next[1] < dec_min))
            if too_close or outside_box:
                continue
            cur = next

            atomics_keep[w] += [(ra, dec, ctime)]

    ra, dec, rap, decp = ({}, {}, {}, {})
    for w in wafer_list:
        ra[w] = np.array([at[0] for at in atomics_dict[w]])
        dec[w] = np.array([at[1] for at in atomics_dict[w]])
        rap[w] = np.array([at[0] for at in atomics_keep[w]])
        decp[w] = np.array([at[1] for at in atomics_keep[w]])

    for iw, w in enumerate(wafer_list):
        c = colors[iw]
        plt.hist(ra[w], alpha=0.3, facecolor=c, label=w)
        plt.hist(rap[w], alpha=0.3, facecolor=c, edgecolor='black', label=f"{w} (keep)")
    plt.title(f"## delta_ra={delta_ra} | delta_dec={delta_dec} ###")
    plt.legend()
    plt.xlabel("RA [deg]")
    plt.savefig(f"{outdir}/ra_hist.png")
    plt.clf()

    for iw, w in enumerate(wafer_list):
        c = colors[iw]
        plt.hist(dec[w], alpha=0.3, facecolor=c, label=w)
        plt.hist(decp[w], alpha=0.3, facecolor=c, edgecolor='black', label=f"{w} (keep)")
    plt.title(f"## delta_ra={delta_ra} | delta_dec={delta_dec} ###")
    plt.legend()
    plt.xlabel("DEC [deg]")
    plt.savefig(f"{outdir}/dec_hist.png")
    plt.clf()

    for iw, w in enumerate(wafer_list):
        c = colors[iw]
        plt.hist(dec[w], alpha=0.3, facecolor=c, label=w)
        plt.hist(decp[w], alpha=0.3, facecolor=c, edgecolor='black', label=f"{w} (keep)")
    plt.title(f"## delta_ra={delta_ra} | delta_dec={delta_dec} ###")
    plt.legend()
    plt.xlabel("DEC [deg]")
    plt.savefig(f"{outdir}/dec_hist.png")
    plt.clf()

    for w in wafer_list:
        plt.title(f"## delta_ra={delta_ra} | delta_dec={delta_dec} ###")
        plt.scatter(ra[w], dec[w], color="b", marker="o",  label="before")
        plt.scatter(rap[w], decp[w], color="r", marker="x", label="after")
        plt.legend()
        plt.xlabel("ra")
        plt.ylabel("dec")
        plt.savefig(f"{outdir}/radec_{w}.png")
        plt.clf()
        plt.close()

        print(f"\n   ### wafer {w} ###")
        print(f"  Reduced number of atomics from {len(atomics)} "
              f"to {len(atomics_keep[w])}")
        print(f"  Before:     ra {np.mean(ra[w])} +- {np.std(ra[w])}")
        print(f"  After:      ra {np.mean(rap[w])} +- {np.std(rap[w])}")
        print(f"  Before:     dec {np.mean(dec[w])} +- {np.std(dec[w])}")
        print(f"  After:      dec {np.mean(decp[w])} +- {np.std(decp[w])}")
    print(f"saving to {outdir}")

    return atomics_keep


def main(args):
    """
    """
    # Load atomics.db
    atomic_db = sqlite_db(args.atomic_db)
    cursor = atomic_db.cursor()

    # Get dictionary with {wafer: (ra, dec, ctime)}
    atomics_dict = {}
    to_query = ["elevation", "azimuth", "ctime"]
    query = f"SELECT {', '.join(to_query)} FROM atomic WHERE freq_channel='{args.freq_channel}'"  # noqa

    #wafer_list = [f"ws{i}" for i in range(7)]
    wafer_list = ["ws0"]

    if args.wafer is not None:
        wafer_list = [args.wafer]
    for wafer in wafer_list:
        print(wafer)
        query_add = f" AND wafer = '{wafer}'"
        res = np.asarray(cursor.execute(query + query_add).fetchall())
        atomics_list = [
            sum([get_radec(float(el), float(az), int(ctime)), (ctime,)], ())
            for el, az, ctime in res
        ]

        # Sort list by right ascension
        atomics_dict[wafer] = sorted(atomics_list, key=lambda tup: tup[0])

    ra_min, ra_max, dec_min, dec_max = (None, None, None, None)
    if args.radec_minmax is not None:
        radec_minmax = args.radec_minmax.strip("[").strip("]")
        ra_min, ra_max, dec_min, dec_max = (float(s) for s in radec_minmax.split(","))  # noqa
        print(f"ra_min = {ra_min}, ra_max = {ra_max}, "
              f"dec_min = {dec_min}, dec_max = {dec_max}, ")

    atomic_list = make_minimal_patch(
        atomics_dict, outdir=args.output_dir,
        delta_ra=args.delta_ra, delta_dec=args.delta_dec,
        ra_min=ra_min, ra_max=ra_max, dec_min=dec_min, dec_max=dec_max
    )

    for wafer in wafer_list:
        new_db = sqlite_db(f"{args.output_dir}/atomic_db_{wafer}.sqlite")
        copy_db_structure(atomic_db, new_db)

        for _, _, ctime in atomic_list[wafer]:
            # TODO: Query atomic_db
            # f"SELECT * from atomic WHERE ctime = '{int(ctime)}' and wafer = '{wafer}'"
            # Then insert resulting column into new_db
            pass

        new_db.close()
        atomic_db.close()
        print(f"{args.output_dir}/atomic_db_{wafer}.sqlite")

    np.savez(
        f"{args.output_dir}/atomic_list.npz", atomic_list=atomic_list
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get atomics list")

    parser.add_argument("--atomic_db", help="Atomic database")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--freq_channel", help="Frequency channel")
    parser.add_argument("--wafer", help="Wafer slot", default=None)
    parser.add_argument(
        "--delta_dec", type=int,
        help="Scales the average distance in DEC between neighboring "
             "atomic maps in the patch."
    )
    parser.add_argument(
        "--delta_ra", type=int,
        help="Scales the average distance in RA between neighboring "
             "atomic maps in the patch."
    )
    parser.add_argument(
        "--radec_minmax",
        help="Delimit sky patch, in format [RA_min,RA_max,DEC_min,DEC_max]",
        type=str
    )

    args = parser.parse_args()

    main(args)
