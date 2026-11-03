import sys
sys.path.append("/shared_home/erosenberg/code/bb-awg-scripts/pipeline/")
sys.path.append("/shared_home/erosenberg/code/bb-awg-scripts/pipeline/bundling")
from configs import Cfg
import bundling_utils
from pixell import enmap
import os, shutil
import sqlite3
import pandas as pd
import numpy as np
import argparse
from sotodlib.utils.procs_pool import get_exec_env

def get_weight(row, shape, wcs, pix_type, map_dir):
    """
    Read weight map. Entries with valid=0 return the scalar value 0.
    row is a single row (pd.Series) of an atomic_db.
    shape, wcs are the geometry of the template output map.
    map_dir is the directory containing the atomics.
    """
    if row.valid == 1:
        fname = bundling_utils.prefixpath2fname(row.prefix_path, map_dir, pix_type=pix_type) % "weights"
        ww = bundling_utils.read_map(fname, pix_type, is_weights=True)
        if pix_type == 'car':
            ww = enmap.extract(ww, shape, wcs)
    else:
        ww = 0
    return ww

def weight_stats(weight):
    """Get statistics from a weight map. Assume T, Q, U."""
    mean_qu = np.mean(weight[1:], axis=0)
    positive = np.where(mean_qu > 0)
    sub_weights = mean_qu[positive]
    if len(sub_weights) == 0:
        return None, None, None
    sumw = np.sum(sub_weights)
    mean = np.mean(sub_weights)
    median = np.median(sub_weights)
    return sumw, mean, median

def _make_science_split(rows, split_pair, shape, wcs, pix_type, map_dir, list_of_avail_splits):
    """Given a dataframe of split rows compute a new one for the science split."""
    r1 = rows.iloc[0]
    weights = []
    for ii, (_, row) in enumerate(rows.iterrows()):
        assert all([r1[col] == row[col] for col in ['obs_id', 'freq_channel', 'wafer']])
        assert row.split_label == sorted(split_pair)[ii]
        weights.append(get_weight(row, shape, wcs, pix_type, map_dir))
    weight = sum(weights)
    sumw, meanw, medianw = weight_stats(weight)
    if sumw is None:
        return None
    new_row = r1.copy()
    replacements = {"split_label": "science", "valid": 0, "prefix_path": '', "total_weight_qu": sumw,
                    "split_detail": ','.join(map(str, list_of_avail_splits)), "mean_weight_qu": meanw, "median_weight_qu": medianw}
    for key, val in replacements.items():
        new_row[key] = val
    return new_row

def make_science_split(rows, config, shape, wcs):
    """Given all rows corresponding to a single obs/wafer/freq, return the science row"""
    split_labels = rows['split_label'].to_list()
    split_pair = config.bundling.coadd_split_pair
    row_pair = rows[rows['split_label'].isin(split_pair)]
    if (len(row_pair) != len(split_pair)) or not ((row_pair.valid==1).any()):
        return None ## Failure
    row_out = _make_science_split(row_pair, split_pair, shape, wcs,
                                  config.pix_type, config.bundling.map_dir, split_labels)
    return row_out

def backup_db(dbname):
    backup_db_name = dbname + ".save"
    if not os.path.exists(backup_db_name):
        shutil.copy2(dbname, backup_db_name)


def main(config, executor, as_completed_callable):
    atomic_db = config.atomic_db
    backup_db(atomic_db)
    conn = sqlite3.connect(atomic_db)
    df = pd.read_sql_query("SELECT * FROM atomic ORDER BY obs_id,freq_channel,wafer,split_label", conn)
    conn.close()

    if df['split_label'].eq('science').any():
        print("Already contains science splits")
        return

    if config.pix_type == 'car':
        shape, wcs = enmap.read_map_geometry(config.car_map_template)
    else:
        shape, wcs = None, None

    rows_out = []
    nmissing = 0
    group_it = df.groupby(['obs_id', 'freq_channel', 'wafer'], sort=False, dropna=False)
    futures = [executor.submit(make_science_split, rows, config, shape, wcs) for _, rows in group_it]
    for future in as_completed_callable(futures):
        row_out = future.result()
        if row_out is None: # Correct slits not available
            nmissing += 1
        else:
            rows_out.append(row_out)

    print(f"Missing {nmissing}")
    science_df = pd.DataFrame(rows_out)
    if len(rows_out) > 0:
        print(f"Writing science rows to {atomic_db}")
        conn = sqlite3.connect(atomic_db)
        science_df.to_sql('atomic', conn, if_exists='append', index=False)
        conn.close()
    else:
        print("No science rows produced")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make science splits")
    parser.add_argument("--config_file", type=str, help="yaml file with configuration.")
    parser.add_argument("--nproc", type=int, default=56, help="Number of parallel processes for concurrent futures.")
    args = parser.parse_args()
    rank, executor, as_completed_callable = get_exec_env(args.nproc)
    if rank == 0:
        config = Cfg.from_yaml(args.config_file)
        main(config, executor, as_completed_callable)
