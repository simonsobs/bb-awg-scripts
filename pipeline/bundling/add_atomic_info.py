from sotodlib import core
import sqlite3, numpy as np
import os
from pixell import enmap
from so3g.proj import coords as so3g_coords
import datetime as dt
import ephem
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import Cfg

def get_site(site_name, ctime):
    site_ = so3g_coords.SITES[site_name].ephem_observer()
    dtime = dt.datetime.fromtimestamp(ctime, dt.timezone.utc)
    site_.date = ephem.Date(dtime)
    return site_
    
def get_distance(obj_name, site_name, ctime, az, el):
    site_ = get_site(site_name, ctime)
    ephem_fn = {"sun": ephem.Sun, "moon": ephem.Moon}
    obj = ephem_fn[obj_name](site_)
    dist = np.degrees(ephem.separation((obj.az, obj.alt), (np.radians(az), np.radians(el))))
    return dist

def get_utc_hour(ctime):
    dtime = dt.datetime.fromtimestamp(ctime, dt.timezone.utc)
    return dtime.hour

def get_radec(ctime, az, el, roll, site):
    #az, el in degrees. roll in *RADIANS*
    csl = so3g_coords.CelestialSightLine.az_el(ctime, az, el, roll, weather='typical', site=site)
    ra, dec, _, _ = csl.coords()[0]  # In radians
    return np.rad2deg(ra), np.rad2deg(dec)

def read_obs_info_db(dbname):
    db = core.metadata.ManifestDb(dbname)
    return db

def prepare_cols(config, columns, force_include=[])
    cols = []
    try:
        props = config.bundle_db_cfg.inter_obs_props
        cols += list(props.keys())
    except AttributeError:
        print("Warning: config.bundle_db_cfg.inter_obs_props does not exist")
    cols += [x for x in force_include if x not in cols]
    all_known_cols = columns['otf'] + columns['obs_info'] + columns['obsdb']
    unknown = ~np.isin(cols, all_known_cols)
    if np.any(unknown):
        raise ValueError(f"Unrecognized columns {cols[unknown]} requested")
    return cols

def get_info(obs_list, col, info_type, cache={}):
    if col in cache:
        return cache[col], cache
    fn = {'otf': get_info_otf, 'obsdb': get_info_obsdb, 'obs_info': get_info_obs_info}[info_type]
    out = fn(obs_list, col)

    if col in ['ra_center', 'dec_center']:
        ra, dec = out
        cache["ra_center"] = ra
        cache["dec_center"] = dec
        out = cache[col]
    return out, cache

def get_info_otf(obs_list, col):
    if col in ['ra_center', 'dec_center']:
        return get_radec_obs(obs_list)
    elif col == 'sun_distance':
        return get_distance_obs('sun', obs_list)
    elif col == 'moon_distance':
        return get_distance_obs('moon', obs_list)
    elif col == 'utc_hour':
        return get_utc_hour_obs(obs_list)
    else:
        raise ValueError(f"{col} can't be calculated OTF")

def get_info_obsdb(obs_list, col):
    
    
    
def main(config, columns, force_include=False):
    input_cols = prepare_cols(config, columns, force_include=force_include)
    cache={}
    for col in input_cols:
        info_type = "otf" if col in columns['otf'] else "obsdb" if col in columns['obsdb'] else "obs_info"
        
        out, cache = get_info(obs_list, col, info_type, cache)
    
    # Read obs_info_db. Check columns.
    # Determine if all requested columns are in otf or obs_info
    # Add columns to atomic that don't exist
    # For each obs:
        # Compute otf or add from obs_info
        # Warn if missing from obs_info

        
if __name__ == "__main__":
    obs_info_db = "/cephfs/soukdata/sat_analysis/iso_data/obs_info/satp1_db_20251217.sqlite"
    parser = argparse.ArgumentParser(description="Add atomic info")
    parser.add_argument("--config_file", type=str, help="yaml file with configuration.")
    args = parser.parse_args()
    config = Cfg.from_yaml(args.config_file)
    
    columns = {'otf': ['sun_distance', 'moon_distance', 'utc_hour', 'ra_center,' 'dec_center'],
               'obsdb': ['scan_speed', 'scan_acc', 'roll_angle'],               
               'obs_info': ['pwv', 'dpwv', 'f_hwp', 'ambient_temperature', 'uv', 'wind_speed', 'wind_direction'],
               }
    if obs_info_db is None:
        columns['obs_info'] = []
    
    overwrite_existing=False  # If column has any non-None values don't change it
    force_include = [] # columns['otf'] + columns['obsbd'] to include all these regardless of config
    main(config, columns, force_include=force_include)
