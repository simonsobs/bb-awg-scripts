import argparse
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


def make_minimal_patch(atomics_dict, outdir, delta_ra, delta_dec,
                       ra_min=None, ra_max=None, dec_min=None, dec_max=None):
    """
    Loops over all atomics, then removes all atomics with (ra, dec) within
    a radius of (delta_ra, delta_dec) within any other map in the list.
    """
    radec = list(atomics_dict.values())

    # Sort 2d coordinates by ascending ra metric
    plus = np.array([r for r, d in radec])
    # obs_ids = [k[0] for k in list(atomics_dict.keys())]
    indexes = np.argsort(plus)
    indexes_keep = []

    # Start at minimum
    cur = radec[indexes[0]]
    # ocur = obs_ids[indexes[0]]

    for idx in indexes[1:]:
        next = radec[idx]
        # onext = obs_ids[idx]
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
        # ocur = onext

        indexes_keep.append(idx)

    atomic_list = [list(atomics_dict.keys())[idx] for idx in indexes_keep]

    print(f"  Reduced number of atomics from {len(indexes)} to {len(indexes_keep)}")  # noqa
    radec_keep = [radec[i] for i in indexes_keep]
    ra = np.array([rd[0] for rd in radec])
    dec = np.array([rd[1] for rd in radec])
    rap = np.array([rd[0] for rd in radec_keep])
    decp = np.array([rd[1] for rd in radec_keep])

    print(f"## delta_ra={delta_ra} | delta_dec={delta_dec} ###")
    plt.hist(ra, alpha=0.3, label="ra")
    plt.hist(rap, alpha=0.3, label="ra_keep")
    plt.savefig(f"{outdir}/ra_hist.png")
    plt.title(f"## delta_ra={delta_ra} | delta_dec={delta_dec} ###")
    plt.legend()
    plt.xlabel("ra [deg]")
    plt.clf()

    plt.hist(dec, alpha=0.3, label="dec")
    plt.hist(decp, alpha=0.3, label="dec_keep")
    plt.title(f"## delta_ra={delta_ra} | delta_dec={delta_dec} ###")
    plt.xlabel("dec [deg]")
    plt.legend()
    plt.savefig(f"{outdir}/dec_hist.png")
    plt.clf()
    plt.close()

    plt.title(f"## delta_ra={delta_ra} | delta_dec={delta_dec} ###")
    plt.scatter(ra, dec, color="b", marker="o",  label="before")
    plt.scatter(rap, decp, color="r", marker="x", label="after")
    plt.legend()
    plt.xlabel("ra")
    plt.ylabel("dec")
    plt.savefig(f"{outdir}/radec.png")
    plt.clf()
    plt.close()

    print(f"  Before:     ra {np.mean(ra)} +- {np.std(ra)}")
    print(f"  After:      ra {np.mean(rap)} +- {np.std(rap)}")
    print(f"  Before:     dec {np.mean(dec)} +- {np.std(dec)}")
    print(f"  After:      dec {np.mean(decp)} +- {np.std(decp)}")

    return atomic_list


def main(args):
    """
    """
    # Load atomics.db
    atomic_db = sqlite3.connect(args.atomic_db)
    cursor = atomic_db.cursor()

    # Get dictionary with {(obs_id, wafer, freq): (el, az)}
    to_query = ["obs_id", "freq_channel", "wafer", "elevation", "azimuth",
                "ctime"]
    query = f"SELECT {', '.join(to_query)} FROM atomic"
    res = np.asarray(cursor.execute(query).fetchall())
    atomics_dict = {
        (o, w, f): get_radec(float(el), float(az), int(ct))
        for o, f, w, el, az, ct in res
    }
    # This is a patch. FIXME: this should be included in the SQL query above.
    if args.freq_channel is not None:
        atomics_dict = {(o, w, f): radec
                        for (o, w, f), radec in atomics_dict.items()
                        if f == args.freq_channel}

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

    # DEBUG
    print("len(atomic_lilst) = {len(atomic_list)}")

    np.savez(
        f"{args.output_dir}/atomic_list.npz", atomic_list=atomic_list
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get atomics list")

    parser.add_argument("--atomic_db", help="Atomic database")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--freq_channel", help="Frequency channel",
                        default=None)
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
