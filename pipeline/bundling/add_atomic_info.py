from sotodlib import core
import sqlite3, numpy as np
import os
from so3g.proj import coords as so3g_coords
import datetime as dt
import ephem
import argparse
import sys
import pandas as pd
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import Cfg

### Utility functions for on-the-fly calculations ###
def get_site(site_name, ctime):
    site_ = so3g_coords.SITES[site_name].ephem_observer()
    dtime = dt.datetime.fromtimestamp(ctime, dt.timezone.utc)
    site_.date = ephem.Date(dtime)
    return site_

def get_distance(obj_name, ctime, az, el, site_name):
    """az, el should be in radians"""
    if np.any(np.abs(np.array([az, el])) > 2*np.pi): raise ValueError("az, el should be in radians")
    site_ = get_site(site_name, ctime)
    ephem_fn = {"sun": ephem.Sun, "moon": ephem.Moon}
    obj = ephem_fn[obj_name](site_)
    dist = np.degrees(ephem.separation((obj.az, obj.alt), (az, el)))
    return dist

def get_utc_hour(ctime):
    dtime = dt.datetime.fromtimestamp(ctime, dt.timezone.utc)
    return dtime.hour

def get_radec(ctime, az, el, site):
    if np.any(np.abs(np.array([az, el])) > 2*np.pi): raise ValueError("az, el should be in radians")
    csl = so3g_coords.CelestialSightLine.az_el(ctime, az, el, weather='typical', site=site)
    ra, dec, _, _ = csl.coords()[0]  # In radians
    return np.rad2deg(ra), np.rad2deg(dec)

### Functions for retrieving info ###
def read_obs_info_db(dbname):
    if dbname is None:
        return []
    else:
        db = core.metadata.ManifestDb(dbname)
        return db.inspect()

def read_sqlite(dbname, query):
    db_conn = sqlite3.connect(dbname)
    db = pd.read_sql_query(query, db_conn)
    db_conn.close()
    return db

def prepare_cols(config, columns, force_include=[], cat_order=['otf', 'obs_info', 'obsdb'], atomic_db=None, overwrite=False):
    """
    Determine which columns need to be added to the atomic db.
    By default it will be the ones listed in config.bundle_db_cfg.inter_obs_props. Anything listed in force_include will also be added.
    cat_order is the order of precedence of the methods in case a col can come from multiple. First is first precedence.
    if overwrite then columns in the existing atomic_db will be overwritten.
    Output is {cat: [col1, col2, ...]} for cat in cat_order.
    """
    if atomic_db is None:
        atomic_db = read_sqlite(config.atomic_db, "SELECT * FROM atomic")
    col_dict = {cat_name: [] for cat_name in cat_order}
    cols = []
    try:
        props = config.bundle_db_cfg.inter_obs_props
        cols += list(props.keys())
    except AttributeError:
        print("Warning: config.bundle_db_cfg.inter_obs_props does not exist")
    cols += [x for x in force_include if x not in cols]
    for col in cols:
        missing = (col not in atomic_db) or (atomic_db[col].values == None).all()
        if (not missing) and (not overwrite):
            print(f"{col} exists, skipping")
            continue
        added = False
        for cat in cat_order:
            if (not added) and (col in columns[cat]):
                col_dict[cat].append(col)
                added = True
        if (not added):
            raise ValueError(f"Unrecognized columns {col} requested")
    return col_dict

def get_info_otf(atomic_db_entry, cols):
    """Get info that can be calculated on the fly"""
    ctime, az, el = atomic_db_entry['ctime'], atomic_db_entry['azimuth'], atomic_db_entry['elevation']
    az, el = np.radians(az), np.radians(el)
    site_name = "so_sat" + atomic_db_entry['telescope'][-1]
    out = {}
    if 'sun_distance' in cols:
        out['sun_distance'] = get_distance('sun', ctime, az, el, site_name)
    if 'moon_distance' in cols:
        out['moon_distance'] = get_distance('moon', ctime, az, el, site_name)
    if 'utc_hour' in cols:
        out['utc_hour'] = get_utc_hour(ctime)
    if 'ra_center' in cols or 'dec_center' in cols:
        out['ra_center'], out['dec_center'] = get_radec(ctime, az, el, site_name)
    return out

def get_info_obsdb(obs_id, cols, context):
    """Get info coming from the obs_db"""
    if not cols:
        return {}
    if context is None:
        raise ValueError(f"Cannot load {cols} from obsdb: no context provided")
    obsdb_renames = {"scan_speed": "scan_vel", "scan_acc": "scan_accel", "roll_angle": "roll_center"}
    ctx = core.Context(context) if type(context) is str else context
    obss = ctx.obsdb.get(obs_id)
    out = {key: obss[obsdb_renames[key]] for key in cols}
    return out

def get_info_obs_info(obs_id, cols, info_entry):
    """Get info coming from an obs_info database"""
    obs_info_renames = {"pwv": "pwv", "dpwv": "dpwv", "scan_acc": "az_acc", "scan_speed": "az_vel", "uv": "uv",
                        "ambient_temperature": "ambient_temp", "wind_speed": "wind_spd", "wind_direction": "wind_dir", "f_hwp": "hwp_rate"}
    out = {key: info_entry[obs_info_renames[key]] for key in cols}
    if "f_hwp" in out:
        out["f_hwp"] = out["f_hwp"] * info_entry["hwp_direction"]
    return out

def add_all_info(obslist, atomic_db, input_cols, context=None, info_db=None):
    entries = read_obs_info_db(info_db)
    obs_ids_infodb = np.array([e['obs:obs_id'] for e in entries])

    for obs_id in obslist:
        oid_inds = np.where(atomic_db['obs_id'] == obs_id)[0]
        atomic_db_entry = atomic_db.iloc[oid_inds[0]]
        info_otf = get_info_otf(atomic_db_entry, input_cols['otf'])
        info_obsdb = get_info_obsdb(obs_id, input_cols['obsdb'], context)
        if len(obs_ids_infodb) > 0:
            id_info = np.argwhere(obs_id == obs_ids_infodb)[0][0]
            info_entry = entries[id_info]
            info_infodb = get_info_obs_info(obs_id, input_cols['obs_info'], info_entry)
        else:
            if len(input_cols['obs_info']) > 0:
                raise ValueError(f"No obs_info: Cannot load {input_cols['obs_info']}")
            info_infodb = {}
        for sub_dict in [info_otf, info_obsdb, info_infodb]:
            for col in sub_dict:
                if col not in atomic_db.columns:
                    atomic_db[col] = None  # Add new col
                icol = np.where(atomic_db.columns == col)[0][0]
                atomic_db.iloc[oid_inds, icol] = sub_dict[col]
    return atomic_db

### Main ###
def add_atomic_info(config, columns, context=None, info_db=None, force_include=[], overwrite=False, cat_order=['otf', 'obs_info', 'obsdb']):
    atomic_db = read_sqlite(config.atomic_db, "SELECT * FROM atomic")
    obs_ids = np.unique(atomic_db['obs_id'].values)
    input_cols = prepare_cols(config, columns, force_include=force_include, cat_order=cat_order, atomic_db=atomic_db, overwrite=overwrite)
    print("Adding: ", input_cols)
    out = add_all_info(obs_ids, atomic_db, input_cols, context=context, info_db=info_db)

    savename = config.atomic_db + ".save"
    if not os.path.exists(savename):
        shutil.copy(config.atomic_db, savename)

    conn = sqlite3.connect(config.atomic_db)
    out.to_sql('atomic', conn, if_exists='replace', index=False)
    conn.close()
    return out

def main():
    parser = argparse.ArgumentParser(description="Add atomic info")
    parser.add_argument("--config_file", type=str, help="yaml file with configuration.")
    parser.add_argument("--context", type=str, default=None, help="Context file")
    parser.add_argument("--obs-info-db", type=str, default=None, help="Obs info db")
    args = parser.parse_args()
    config = Cfg.from_yaml(args.config_file)
    columns = {'otf': ['sun_distance', 'moon_distance', 'utc_hour', 'ra_center', 'dec_center'],
               'obsdb': ['scan_speed', 'scan_acc', 'roll_angle'],
               'obs_info': ['pwv', 'dpwv', 'f_hwp', 'ambient_temperature', 'uv', 'wind_speed', 'wind_direction'],
               }

    overwrite = False  # If column has any non-None values don't change it
    force_include = [] # columns['otf']+columns['obsdb']+columns['obs_info']

    add_atomic_info(config, columns, args.context, args.obs_info_db, force_include, overwrite=overwrite)

if __name__ == "__main__" or True:
    main()
