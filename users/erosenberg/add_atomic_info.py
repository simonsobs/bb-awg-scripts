from sotodlib import core
import sqlite3, numpy as np
import os
from pixell import enmap
from so3g.proj import coords as so3g_coords
import datetime as dt
import ephem


def get_col_names(con, table_name):
    """Get column names from an sqlite table"""
    query = f"SELECT * FROM {table_name}"
    if isinstance(con, str):
        con = sqlite3.connect(con)
        cur = con.execute(query)
        con.close()
    else:
        cur = con.execute(query)
    desc = cur.description
    return [entry[0] for entry in desc]

# First, let's add missing columns to atomic db
def add_missing_cols(atomic_db_name, new_column_dict):
    """ new_column_dict = {'dpwv': 'FLOAT', ...}"""
    con = sqlite3.connect(atomic_db_name)
    cols = get_col_names(con, "atomic")
    cur = con.cursor()
    for col in new_column_dict:
        if col not in cols:
            cur.execute("""ALTER TABLE atomic ADD COLUMN "{}" {}""".format(col, new_column_dict[col]))
    con.commit()
    con.close()

def add_info(obs_info_db, atomic_db, site):
    db = core.metadata.ManifestDb(obs_info_db)
    entries = db.inspect()
    obs_ids_1 = [e['obs:obs_id'] for e in entries]
    obs_ids_1 = np.array(obs_ids_1)

    # I need a list of obs_id in my atomic db
    con = sqlite3.connect(atomic_db)
    cur = con.cursor()

    cur.execute("SELECT obs_id FROM atomic")
    obs_id_list = cur.fetchall()
    obs_id_list = np.unique(obs_id_list)

    # We also need all rows to extract az and el for sun distance calculation
    cur.execute("SELECT * FROM atomic ")
    rows = cur.fetchall()
    rows = np.array(rows)

    # now I loop over the obs_id in the atomic db, updating values

    site_ = so3g_coords.SITES[site].ephem_observer()

    for obs_id in obs_id_list:
        idx = np.argwhere(obs_ids_1 == obs_id)[0][0]

        pwv = entries[idx]['pwv']
        dpwv = entries[idx]['dpwv']
        scan_acc = entries[idx]['az_acc']
        scan_speed = entries[idx]['az_vel']
        uv = entries[idx]['uv']
        ambient_temperature = entries[idx]['ambient_temp']
        f_hwp = entries[idx]['hwp_direction'] * entries[idx]['hwp_rate']


        # we also need to calculate the sun distance
        idx_rows = np.where(rows[:,0] == obs_id)[0][0]
        ctime = rows[idx_rows,4]
        az = rows[idx_rows,10]
        el = rows[idx_rows,9]
        site_ = so3g_coords.SITES[site].ephem_observer()
        dtime = dt.datetime.fromtimestamp(ctime, dt.timezone.utc)
        site_.date = ephem.Date(dtime)
        sun = ephem.Sun(site_)
        sun_distance = np.degrees(ephem.separation((sun.az, sun.alt), (np.radians(az), np.radians(el))))


        # we also need to calculate the moon distance
        idx_rows = np.where(rows[:,0] == obs_id)[0][0]
        ctime = rows[idx_rows,4]
        az = rows[idx_rows,10]
        el = rows[idx_rows,9]
        dtime = dt.datetime.fromtimestamp(ctime, dt.timezone.utc)
        site_.date = ephem.Date(dtime)
        moon = ephem.Moon(site_)
        moon_distance = np.degrees(ephem.separation((moon.az, moon.alt), (np.radians(az), np.radians(el))))

        # now we update

        cur.execute("UPDATE atomic SET pwv = %f WHERE obs_id = '%s' " % (pwv, obs_id))
        cur.execute("UPDATE atomic SET dpwv = %f WHERE obs_id = '%s' " % (dpwv, obs_id))
        cur.execute("UPDATE atomic SET scan_speed = %f WHERE obs_id = '%s' " % (scan_speed, obs_id))
        cur.execute("UPDATE atomic SET scan_acc = %f WHERE obs_id = '%s' " % (scan_acc, obs_id))
        cur.execute("UPDATE atomic SET sun_distance = %f WHERE obs_id = '%s' " % (sun_distance, obs_id))
        cur.execute("UPDATE atomic SET uv = %f WHERE obs_id = '%s' " % (uv, obs_id))
        cur.execute("UPDATE atomic SET f_hwp = %f WHERE obs_id = '%s' " % (f_hwp, obs_id))
        cur.execute("UPDATE atomic SET ambient_temperature = %f WHERE obs_id = '%s' " % (ambient_temperature, obs_id))

        cur.execute("UPDATE atomic SET moon_distance = %f WHERE obs_id = '%s' " % (moon_distance, obs_id))

    con.commit()
    con.close()

def main():
    tel = "satp3"
    new_columns = {"dpwv":"FLOAT", "scan_acc":"FLOAT", "ambient_temperature":"FLOAT", "uv":"FLOAT", "moon_distance":"FLOAT"}
    atomic_db_name = f"/scratch/gpfs/SIMONSOBS/sat-iso/v2/mapmaking/{tel}_20250507_nr/atomic_db.sqlite.updated"
    obs_info_db = f'/home/ms3067/shared_files/iso/obs_info/250123_obs_info/{tel}/db.sqlite'
    site = 'so_sat3' if tel == 'satp3' else 'so_sat1'

    # First, let's add missing columns to atomic db
    add_missing_cols(atomic_db_name, new_columns)

    # Add the info
    add_info(obs_info_db, atomic_db_name, site)


if __name__ == '__main__':
    main()
