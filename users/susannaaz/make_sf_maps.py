import os
import numpy as np
import pandas as pd
import sqlite3 as sq
import warnings
from concurrent.futures import ProcessPoolExecutor
from sotodlib import coords
from sotodlib.io.metadata import read_dataset, write_dataset
from sotodlib.core.metadata import ResultSet
from pixell import enmap, utils
import healpy as hp
from concurrent.futures import ProcessPoolExecutor
from sotodlib.coords import helpers

from query_atomic_db import AtomicDB
from fits_utils import save_to_fits
import re
import random

warnings.simplefilter('ignore')

def extract_key(path, split="full"):
    match = re.search(f"atomic_\d+_ws\d+_f\d+_{split}", path)
    return match.group(0) if match else None

def wrap_inv_var(df: pd.DataFrame) -> None:
    """Calculate and add the inverse variance for each observation in the result set."""
    # Get atomic maps database 
    idir = "/scratch/gpfs/SIMONSOBS/users/sa5705/SO/maps/"
    db_name = "atomic_maps.db" 
    db_path = os.path.join(idir, db_name)
    if os.path.exists(db_path):
        db = AtomicDB(idir, db_name)
        atomic_df = db.query_database()
        
        ## Extract maps_id and add key to df and atomic_df
        atomic_df["map_id"] = atomic_df["prefix_path"].apply(extract_key)
        df["map_id"] = df["input_file"].apply(extract_key)

        ## Merge the DataFrames
        ## i.e. keep in atomic_df only the rows with the overlapping maps_id 
        ## that are present in df_box12
        filtered_df = atomic_df[atomic_df['map_id'].isin(df['map_id'])]

        ## Calculate inverse variance
        if "mean_weight_qu" in filtered_df.keys():
            inv_var = 1 / filtered_df["mean_weight_qu"]
        else:
            raise ValueError(f"The database file {db_path} does not contain the needed info.")
    else:
        hits_files = np.array(df['input_file'])
        weight_files = np.array([path.replace("hits", "weights") for path in hits_files])
        inv_var = []
        for wfile, hfile in zip(weight_files, hits_files):
            w_ = enmap.read_map(wfile)  # Weight map
            h_ = enmap.read_map(hfile)   # Hits map
            var = (np.nanmean(w_[1]) + np.nanmean(w_[2])) / 2
            ivar = 1/var
            inv_var.append(ivar)
    add_column(df, 'inv_var', inv_var)


def add_column(dataframe: pd.DataFrame, column_name: str, new_column_data: np.ndarray) -> None:
    """
    Add a new column to a DataFrame.
    
    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to which the column will be added.
    - column_name (str): The name of the new column.
    - new_column_data (np.ndarray): The data for the new column.
    
    Raises:
    - ValueError: If the new column's length does not match the DataFrame's length or if the column name already exists.
    """
    if len(dataframe) != len(new_column_data):
        raise ValueError("The length of the new column does not match the number of rows in the DataFrame.")

    if column_name in dataframe.columns:
        raise ValueError(f"{column_name} is already in the DataFrame.")
    
    # Add the new column to the DataFrame
    dataframe[column_name] = new_column_data

    
def filter_by_box(idir: str, 
                  box: list, 
                  buffer_deg_dec: float = 5, 
                  buffer_deg_ra: float = 10) -> pd.DataFrame:
    """
    Filter the DataFrame based on a bounding box with an optional buffer.
    
    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing celestial coordinates with 'dec_centre' and 'ra_centre'.
    - box (list): A list containing two tuples/lists defining the bounding box corners
                  in the format [[dec_min, ra_min], [dec_max, ra_max]].
    - buffer_deg_dec (float): Buffer in degrees for declination.
    - buffer_deg_ra (float): Buffer in degrees for right ascension.
    
    Returns:
    - pd.DataFrame: A filtered DataFrame containing only the rows within the specified box.
    """
    buffer_dec = np.deg2rad(buffer_deg_dec)
    buffer_ra = np.deg2rad(buffer_deg_ra)
    
    db_name = "centre_RA_DEC_f090_info.db"
    db = AtomicDB(idir, db_name)
    query = f"ra_centre > '{box[0][1] - buffer_ra}' AND ra_centre < '{box[1][1] - buffer_ra}' \
             AND dec_centre > '{box[0][0] - buffer_dec}' AND dec_centre < '{box[1][0] + buffer_dec}'"
    atomic_df = db.query_database(query=query) #table_name)
    return atomic_df


def write_db_including_ivar(df: pd.DataFrame, db_file: str) -> None:
    """Write the DataFrame to an SQLite database, including inverse variance."""
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    
    with sq.connect(db_file) as conn:
        df.to_sql('results', conn, if_exists='append', index=False)
        print(f"DataFrame written to {db_file} successfully.")

    # Convert RA and DEC from radians to degrees
    ra_deg = np.rad2deg(dataframe['ra_centre'])
    dec_deg = np.rad2deg(dataframe['dec_centre'])
    
    scatter = ax.scatter(ra_deg, dec_deg, s=10, **kwargs)  # Set size of scatter points to 10

    return

def get_location_dict(dataframe: pd.DataFrame, 
                      ra_bin_edges: np.ndarray, 
                      dec_bin_edges: np.ndarray, 
                      num_obs_threshold: int) -> dict:
    """
    Generate a dictionary of filtered DataFrame segments based on RA and Dec bin edges.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame containing 'ra_centre' and 'dec_centre' columns in radians.
    - ra_bin_edges (list or np.ndarray): The bin edges for Right Ascension (RA) in degrees.
    - dec_bin_edges (list or np.ndarray): The bin edges for Declination (Dec) in degrees.
    - num_obs_threshold (int, optional): The minimum number of observations required to include a segment in the dictionary. Default is 8.

    Returns:
    dict: A dictionary where keys are tuples of (ra_min, ra_max, dec_min, dec_max) and values are DataFrame segments that meet the observation threshold.
    """
    location_dict = {}
    for ra_min, ra_max in zip(ra_bin_edges[:-1], ra_bin_edges[1:]):
        for dec_min, dec_max in zip(dec_bin_edges[:-1], dec_bin_edges[1:]):
            mask = (
                (ra_min < np.rad2deg(dataframe['ra_centre'])) & (np.rad2deg(dataframe['ra_centre']) < ra_max) &
                (dec_min < np.rad2deg(dataframe['dec_centre'])) & (np.rad2deg(dataframe['dec_centre']) < dec_max)
            )
            filtered_df = dataframe[mask]
            if len(filtered_df) >= num_obs_threshold:
                location_dict[(ra_min, ra_max, dec_min, dec_max)] = filtered_df
    return location_dict


def get_signs_for_location_dict(location_dict: dict, 
                                n_mc: int = 8, 
                                inner_product_threshold: float = 0.6) -> None:
    """
    Calculate and add signs for each box in the location dictionary.
    
    Parameters:
    - location_dict (dict): Dictionary containing boxes with observation data.
    - n_mc (int): Number of Monte Carlo realizations.
    - inner_product_threshold (float): Threshold for inner product calculations.
    """
    #print(location_dict.keys())

    for box_name, val in location_dict.items():
        # Assuming val is a DataFrame with an 'inv_var' column
        signs_array = get_sign_flip_realizations_array(
            obs_weights=val['inv_var'], n_mc=n_mc,
            inner_product_threshold=inner_product_threshold
        )

        # Add each realization as a new column in the DataFrame
        for i_mc in range(n_mc):
            val[f'signs_{i_mc:03}'] = signs_array[i_mc]

            
def get_signs(obs_weights: np.ndarray,
              seed = None) -> np.ndarray:
    """Generate random signs based on observation weights."""
    obs_weights /= max(obs_weights)
    obs_weights = np.where(np.isnan(obs_weights), random.random(), obs_weights)
    if seed is not None:
        np.random.seed(seed)
        
    if not all(0 <= w <= 1 for w in obs_weights):
        raise ValueError("All weights must be between 0 and 1.")
    
    # Print weights and random numbers for debugging
    signs = []
    for w in obs_weights:
        rand_num = random.random()
        signs.append(1 if rand_num < w else -1)
    return signs


def get_sign_flip_realizations_array(obs_weights: pd.Series, 
                                     n_mc: int = 32, 
                                     inner_product_threshold: float = 0.5,
                                     seed = None) -> np.ndarray:
    """
    Generate an array of sign flip realizations that are independent of each other.

    Parameters:
    - obs_weights (pd.Series): Series containing observation weights.
    - n_mc (int): Number of Monte Carlo realizations to generate.
    - inner_product_threshold (float): Threshold for inner product calculations.

    Returns:
    - np.ndarray: A 2D array of shape (n_mc, n_obs) containing independent sign flip realizations.

    Raises:
    - ValueError: If unable to find enough independent realizations.
    """
    n_obs = len(obs_weights)
    signs_array = np.zeros((n_mc, n_obs))
    obs_weights = np.array(obs_weights)
    i = 0
    for i_try in range(n_obs*100):
        # Generate new random signs
        _signs_new = get_signs(obs_weights)  # Convert Series to numpy array
        signs_inner_product_checker = np.mean(_signs_new)

        # Check inner products against the threshold
        if np.abs(np.max(signs_inner_product_checker)) < inner_product_threshold:
            signs_array[i] = _signs_new
            i += 1
        if i == n_mc:
            break
    if i < n_mc:
        print(i, n_mc)
        raise ValueError('Could not find sign flip realizations which are independent of each other')
    else:
        return signs_array
    
def combine_location_dict(location_dict: dict) -> pd.DataFrame:
    """
    Combine DataFrames from a location dictionary into a single DataFrame.

    Parameters:
    - location_dict (dict): Dictionary containing boxes with DataFrame entries.

    Returns:
    - pd.DataFrame: A single DataFrame combining all DataFrames from the location dictionary.
    """
    df_list = []  # List to collect DataFrames

    for box_name, val in location_dict.items():
        df_list.append(val)

    # Concatenate all DataFrames in the list into a single DataFrame
    df_combined = pd.concat(df_list, ignore_index=True)

    return df_combined


def dataframe_to_resultset(df: pd.DataFrame) -> ResultSet:
    """
    Convert a pandas DataFrame into a ResultSet.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        ResultSet: A resultset populated from the DataFrame.
    """
    # Create a ResultSet with column names as keys
    rs = ResultSet(keys=df.columns.tolist())
    
    # Append each row of the DataFrame as a tuple
    for _, row in df.iterrows():
        rs.rows.append(tuple(row))

    return rs


def read_and_combine_maps(
    files, 
    signs=None, 
    address=None, 
    show_status=False
) -> np.ndarray:
    """Read and combine maps fits files, applying optional signs to the data."""
    
    if show_status:
        print(f'Combining {len(files)} files: [{os.path.basename(files[0])}, ..., {os.path.basename(files[-1])}]')

    # Default signs to ones if not provided
    signs = np.ones(len(files), dtype=float) if signs is None else signs

    shape, wcs = enmap.fullsky_geometry(res=10 * utils.arcmin, proj="car", variant="CC")
    sum_data = None

    for sign, file in zip(signs, files):
        data = sign * enmap.extract(enmap.read_map(file), shape, wcs)
        sum_data = data if sum_data is None else sum_data + data

    return np.array(sum_data)


def diag_to_mat_weight(weight_diag):
    """Convert a diagonal weight array to a full 3x3 weight matrix."""
    
    # Initialize a 3x3 weight matrix with zeros
    weight = enmap.zeros(shape=(3, 3, *weight_diag.shape[1:]), wcs=weight_diag.wcs)
    # Set diagonal elements manually
    for i in range(3):
        weight[i, i, :, :] = weight_diag[i, :, :]

    return weight


def combine_maps(
    weighted_map_files, 
    weight_files, 
    hits_files,
    process_split=32, 
    signs=None, 
    address=None, 
    show_status=False
) -> tuple:
    """
    Combine weighted maps and weight files.

    Returns:
        tuple: (map, weighted_map, weight_diag)
    """
    
    process_split = min(process_split, len(weighted_map_files))
    weighted_map_files_split = np.array_split(weighted_map_files, process_split)
    weight_files_split = np.array_split(weight_files, process_split)
    hits_files_split = np.array_split(hits_files, process_split)
    
    # Initialize signs if not provided
    signs = np.ones(len(weighted_map_files), dtype=float) if signs is None else signs
    signs_split = np.array_split(signs, process_split)

    with ProcessPoolExecutor(max_workers=process_split) as executor:
        futures_weighted_map = [
            executor.submit(read_and_combine_maps, _weighted_map_files, _signs, address, show_status)
            for _weighted_map_files, _signs in zip(weighted_map_files_split, signs_split)
        ]
        futures_weight = [
            executor.submit(read_and_combine_maps, _weight_files, None, address, show_status)
            for _weight_files in weight_files_split
        ]
        futures_hits = [
            executor.submit(read_and_combine_maps, _hits_files, None, address, show_status)
            for _hits_files in hits_files_split
        ]
        
        weighted_map = sum(future.result() for future in futures_weighted_map)
        weight_diag = sum(future.result() for future in futures_weight)
        hits_map = sum(future.result() for future in futures_hits)
        
    wcs = enmap.read_map(weighted_map_files[0]).wcs
    weighted_map = enmap.enmap(weighted_map, wcs=wcs)
    weight_diag = enmap.enmap(weight_diag, wcs=wcs)
    hits_map = enmap.enmap(hits_map, wcs=wcs)

    weight = diag_to_mat_weight(weight_diag)

    inverse_weight = helpers._invert_weights_map(weight)
    map_result = helpers._apply_inverse_weights_map(inverse_weight, weighted_map)

    return map_result, weighted_map, weight_diag, hits_map



def main(freq, idir):
    
    print("Divide observations by boxes")
    # Read database
    # Define "boxes", i.e. masks
    box1 = np.deg2rad([[-70, -80], [-10, 60]])
    box2_1 = np.deg2rad([[-30, 150], [10, 180]])
    box2_2 = np.deg2rad([[-30, -180], [10, -110]])
    
    # Find observations in box
    idir_db = f"{idir}/maps_info/"
    box1 = filter_by_box(idir_db, box1)
    box2_1 = filter_by_box(idir_db, box2_1)
    box2_2 = filter_by_box(idir_db, box2_2)
    # Combine all boxes
    box12 = pd.concat([box1, box2_1, box2_2]).drop_duplicates().reset_index(drop=True)
    # Add inverse variance and write db
    ivar_db_path = f'{idir_db}/centres_ivar_{freq}.db'
    wrap_inv_var(box12)
    
    # Make dictionary of locations based on RA and DEC bin edges
    ra_bin_edges = np.arange(-180, 181, 20)
    dec_bin_edges = np.arange(-90, 91, 30)
    location_dict = get_location_dict(box12, ra_bin_edges, dec_bin_edges, num_obs_threshold=8)

    print("Get random signs")
    # Get random signs ##TODO: to improve based on weights
    n_mc = 300 
    # Add signs to to df
    get_signs_for_location_dict(location_dict, n_mc=n_mc, inner_product_threshold = 0.6)
    res = combine_location_dict(location_dict)

    signs_mat = []
    for i_mc in range(n_mc):
        signs_mat.append(res[f'signs_{i_mc:003}'])
    signs_mat = np.array(signs_mat)
    
    # Save as resultset
    ress = dataframe_to_resultset(res)
    print(f"Save signs df in {idir}/maps_info/signs_{freq}_info.hdf")
    write_dataset(ress, filename=os.path.join(idir, f'maps_info/signs_{freq}_info.hdf'), 
                  address='signs', overwrite=True)

    # Output sign-flip maps
    hits_files = np.array(res['input_file'])
    weight_files = np.array([path.replace("hits", "weights") for path in hits_files])
    wmap_files = np.array([path.replace("hits", "wmap") for path in hits_files])
    save_dir = os.path.join(idir, 'signflip', freq)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save sign-flip maps in {save_dir} ")
    for i_mc in np.arange(n_mc):
        print(i_mc)
        map_sf, _, _, hits_map_sf = combine_maps(wmap_files, weight_files, hits_files, signs=res[f'signs_{i_mc:03}'])
        sf_map_path = os.path.join(save_dir, f'sf_map_{i_mc:03}.fits')
        if not os.path.exists(sf_map_path):
            save_to_fits(map_sf, f'sf_map_{i_mc:03}', sf_map_path)
            save_to_fits(hits_map_sf, f'sf_hits_{i_mc:03}', sf_map_path)

 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--freq", 
                        type=str, 
                        default='f090', 
                        help="Frequency (default: 'f090')")
    parser.add_argument("--idir", 
                        type=str, 
                        default="/pscratch/sd/s/susannaz/SO_ISO/satp3/analysis/outputs/", 
                        help="Input directory (default: '/pscratch/sd/s/susannaz/SO_ISO/satp3/analysis/outputs/')")

    args = parser.parse_args()
    main(freq=args.freq, idir=args.idir)
