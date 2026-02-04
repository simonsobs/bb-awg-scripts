import os
import numpy as np
from pixell import enmap, enplot
import healpy as hp
from astropy.io import fits
import h5py
import re
from copy import deepcopy
from matplotlib import pyplot as plt

from typing import Optional
from dataclasses import dataclass
import yaml
import pandas as pd

##############################################################################
## Map Operations ##
##############################################################################
def read_map(map_file, pix_type='hp', fields_hp=None, nest_hp=False,
             convert_K_to_muK=False, geometry=None, is_weights=False):
    """
    Read a map from a file, which can be either in HEALPix or CAR format.

    Parameters
    ----------
    map_file: str
        Map file name.
    pix_type: str, optional
        Pixellization type.
    fields_hp: tuple, optional
        Fields to read from a HEALPix map.
    nest_hp: boolean
        Optional; whether to assume nested ording for HEALPix maps.
    convert_K_to_muK: bool, optional
        Convert K to muK.
    geometry: enmap.geometry, optional
        Enmap geometry.
    is_weights: bool
        If map weights are to be read, then make sure to only load the
        TT, QQ, UU map weights, i.e., ignore cross-field weights.

    Returns
    -------
    map_out: np.ndarray
        Loaded map.
    """
    conv = 1
    if convert_K_to_muK:
        conv = 1.e6
    _check_pix_type(pix_type)
    if pix_type == 'hp':
        if is_weights:
            # If num_fields == 9 then we are reading a TQU pixel weights matrix
            num_fields = fits.getheader(map_file, 1)['TFIELDS']
            # Read only TT, QQ, UU weights
            fields_hp = (0, 4, 8) if (num_fields == 9) else (0, 1, 2)
        # Default to loading all fields (field = None) if fields_hp
        # is not provided
        kwargs = {"field": fields_hp, }  # if fields_hp is not None else {}
        m = hp.read_map(map_file, **kwargs)
    else:
        m = enmap.read_map(map_file, geometry=geometry)
        if is_weights:
            # Read only TT, QQ, UU weights
            if m.ndim > 3 and m.shape[0] == m.shape[1]:
                m = np.moveaxis(m.diagonal(), -1, 0)

    return conv*m


def write_map(map_file, map, dtype=None, pix_type='hp',
              convert_muK_to_K=False):
    """
    Write a map to a file, regardless of the pixellization type.

    Parameters
    ----------
    map_file : str
        Map file name.
    map : np.ndarray
        Map to write.
    dtype : np.dtype, optional
        Data type.
    pix_type : str, optional
        Pixellization type.
    convert_muK_to_K : bool, optional
        Convert muK to K.
    """
    if convert_muK_to_K:
        map *= 1.e-6
    _check_pix_type(pix_type)
    if pix_type == 'hp':
        hp.write_map(map_file, map, overwrite=True, dtype=dtype)
    else:
        enmap.write_map(map_file, map)

def write_maps(out_fname, pix_type, bundled_map, weights_map, hits_map=None, fnames=None, dtype=np.float64):
    """
    Save map, weights, hits, and optionally filenames going into bundles.

    Parameters
    ----------
    out_fname: str
        File path to write to. Should have a single 'format' field for
        map type ('map', 'weights', 'hits').
    pix_type: str
        'hp' or 'car'
    fnames: list[str]
        List of file names going into a bundle.
    dtype: type
        dtype for Healpix
    """
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    if fnames is not None:
        out_filenames = out_fname[:out_fname.find(".fits")] + ".txt"
        np.savetxt(out_filenames.format("fnames"), fnames, fmt='%s')
    for imap, tag in zip([bundled_map, weights_map, hits_map], ["map", "weights", "hits"]):
        if imap is not None:
            write_map(out_fname.format(tag), imap, dtype=dtype, pix_type=pix_type)

def read_hdf5_map(fname, to_nest=False):
    """
    Read a HEALPix map in hdf5 format.
    """
    f = h5py.File(fname, "r")
    dset = f["map"]
    header = dict(dset.attrs)

    if header["ORDERING"] == "NESTED":
        file_nested = True
    elif header["ORDERING"] == "RING":
        file_nested = False

    if file_nested and not to_nest:
        mapdata = hp.reorder(dset[:], n2r=True)
    elif not file_nested and to_nest:
        mapdata = hp.reorder(dset[:], r2n=True)
    else:
        mapdata = dset

    return mapdata

def write_hdf5_map(fname, nside, dict_maps, list_of_obsid,
                   nest_or_ring='RING'):
    """
    Write a HEALPix map into hdf5 format.
    """
    with h5py.File(fname, 'w') as f:
        f.attrs['NSIDE'] = nside
        f.attrs['ORDERING'] = nest_or_ring
        f.attrs['OBS_ID'] = list_of_obsid
        for k, v in dict_maps.items():
            f.create_dataset(k, data=v)

def plot_map(out_fname, pix_type, imap, unit_fac=1, vrange=None):
    """Make and save plots of maps. Maps should be single component."""
    if pix_type == "car":
        vmin = -vrange if vrange is not None else None
        plot = enplot.plot(imap*unit_fac, colorbar=True,
                           min=vmin, max=vrange, ticks=10, downgrade=2)
        if out_fname[-4:] in ['.png', '.jpg', '.pdf']:
            out_fname = out_fname[:-4]  # Remove file extensions for enplot
        enplot.write(out_fname, plot)

    elif pix_type == "hp":
        hp.mollview(imap * unit_fac, cmap="RdYlBu_r",
                    min=vmin, max=vrange)
        if out_fname[-4:] not in ['.png', '.jpg', '.pdf']:
            out_fname += ".png"  # Add png if no extension
        plt.savefig(out_fname)
        plt.close()

##############################################################################
## Bundling / Coaddition utility functions #
##############################################################################
def coadd_maps(maps_list, weights_list, hits_list=None, sign_list=None,
               pix_type="hp", res_car=5., car_template_map=None,
               dec_cut_car=None, fields_hp=None, abscal=1, parallelizor=None):
    """
    Coadd a list of weighted maps, a list of map weights, and
    (optionally) a list of hits maps corresponding to a set of atomics.
    Optionally, multiply weighted maps with a sign (+/-1) before coadding.

    Parameters
    ----------
    maps_list: list
        List of weighted TQU maps (string filenames, or numpy arrays).
    weights_list: list
        List of TQU map weights (string filenames, or numpy arrays).
    hits_list: list
        Optional; list of hits maps (string filenames, or numpy arrays).
        If None, ignore.
    sign_list:
        Optional; list of signs (+1 or -1) to multiply weighted maps with.
        If None, do not change maps at all.
    pix_type: str
        Pixelization assumed for input maps, either "hp" for HEALPix or
        "car" for CAR pixelization. Default: "hp"
    res_car: int
        Optional; pixel resolution of input CAR maps in arcminutes.
    car_template_map: str
        Filename from which to get CAR geometry
    dec_cut_car: tuple of int
        Optional; lower and upper declination cut in degrees, assumed
        for the inputCAR maps. Defaults to SAT-compatible sky region.
    fields_hp: tuple
        Tuple of the indexes for the desired fields in the healpix wmap
    abscal: array
        Multiplicative factor for each map. Output maps will be multiplied
        by this number; the weights map will get abscal**-2
    parallelizor: tuple
        (MPICommExecutor or ProcessPoolExecutor, as_completed_callable, num_workers)

    Returns
    -------
    map_coadd: np.array
        Coadded data TQU map, multiplied by the inverse coadded map weights.
    weights_coadd: np.array
        Coadded data TQU weights.
    hits_coadd: np.array
        Optional; coadded hits map. Will only be returned if hits_list given.
    """
    assert len(maps_list) == len(weights_list)
    if sign_list is None:
        sign_list = 1
    elif isinstance(sign_list, list):
        sign_list = np.array(sign_list)
    if isinstance(abscal, list):
        abscal = np.array(abscal)

    sum_fn = _make_parallel_proc(sum_maps, parallelizor) if parallelizor is not None else sum_maps

    if pix_type == "car":
        template = _get_map_template_car(car_template_map, res_car,
                                         dec_cut_car)
    elif pix_type == "hp":
        template = _get_map_template_hp(maps_list[0])

    # Assume multiplicative abscal A: map_cal = map_uncal*A.
    # Then wmap_cal = wmap_uncal*A**-1 and weights_cal = weights_uncal*A**-2
    map_coadd = sum_fn(
        maps_list,
        template,
        pix_type,
        mult=sign_list*abscal**-1,
        fields_hp=fields_hp
    )
    weights_coadd = sum_fn(
        weights_list,
        template,
        pix_type,
        mult=abscal**-2,
        fields_hp=fields_hp,
        is_weights=True
    )
    if hits_list is not None:
        hits_coadd = sum_fn(hits_list, template[0], pix_type)

    good_weights = weights_coadd > 0
    map_coadd[good_weights] /= weights_coadd[good_weights]

    if hits_list is not None:
        return map_coadd, weights_coadd, hits_coadd
    else:
        return map_coadd, weights_coadd

def sum_maps(filenames, template, pix_type, mult=1, condition=lambda x: True,
             islice=slice(None), **read_map_kwargs):
    """Coadd CAR or healpix maps

    Parameters
    ----------
    filenames: list
        List of filenames to read and coadd. Can also be pre-loaded maps.
    template: array
        An empty map to use as a template.
    pix_type: str
        Pixelization, 'car' or 'hp'
    mult: array
        Single number or len(filenames) array of numbers.
        Each map will be multiplied by this.
    condition: function
        Boolean or array(bool) function with an individual map as input.
        If any False this map will not be included in the coadd
    islice: slice
        1d slice to apply to filenames and mult. Used in parallelization.
    read_map_kwargs:
        kwargs passed through to read_map

    Returns
    -------
    out: array
        enmap.ndmap or np.ndarray of the output coadded map
    """

    out = template.copy() * 0
    for ifn, fn in enumerate(filenames[islice]):
        if isinstance(fn, str):
            imap = read_map(fn, pix_type=pix_type, **read_map_kwargs)
        else:
            imap = fn
        imult = mult if np.isscalar(mult) else mult[islice][ifn]
        imap *= imult
        if np.any(np.isnan(imap)):
            print(f"WARNING: nans in {fn}. Skipping.")
        if np.all(condition(imap)):
            _add_map(imap, out, pix_type)
    return out

def coadd_bundles(template, sum_vals, pix_type, coadd_hits=True, coadd_fnames=False,
                  savename=None, **read_map_kwargs):
    """Add bundled maps together. Maps are assumed to be unweighted and
    the same shape/geometry

    Parameters
    ----------
    template: str
        Formattable string filename with args value_to_be_summed
        (bundle id, split id, etc), map_type (map, weights, hits)
    sum_vals: iterable
        List of strings to put into the template for the files to be summed
    pix_type: str
        hp or car
    coadd_hits: bool
        If True also add/return hits
    coadd_fnames: bool
        If True also read the fnames files and add them together
    savename: str
        Formattable string filename with args map_type
    read_map_kwargs: dict
        kwargs passed through to read_map
    """
    out = {'map': None, 'weights': None, 'hits': None, 'filenames': None}
    for val in sum_vals:
        imap = read_map(template.format(val, 'map'), pix_type=pix_type,
                        **read_map_kwargs)
        weights = read_map(template.format(val, 'weights'), pix_type=pix_type,
                           **read_map_kwargs)
        if out['map'] is None:
            out['map'] = imap * weights
            out['weights'] = weights
        else:
            out['map'] += imap * weights
            out['weights'] += weights
        if coadd_hits:
            hits = read_map(template.format(val, 'hits'), pix_type=pix_type,
                            **read_map_kwargs)
            if out['hits'] is None:
                out['hits'] = hits
            else:
                out['hits'] += hits
        if coadd_fnames:
            filenames_fn = template[:template.find(".fits")] + ".txt"
            filenames_fn = filenames_fn.format(val, 'fnames')
            filenames = np.loadtxt(filenames_fn, dtype=str)
            if out['filenames'] is None:
                out['filenames']  = filenames
            else:
                out['filenames'] = np.concatenate([out['filenames'], filenames])

    del imap
    wmap, weights = out['map'], out['weights']
    good = weights > 0
    wmap[good] /= weights[good]

    if savename is not None:
        write_map(savename.format("map"), wmap, dtype=wmap.dtype,
                  pix_type=pix_type)
        write_map(savename.format("weights"), weights, dtype=weights.dtype,
                  pix_type=pix_type)
        if coadd_hits:
            write_map(savename.format("hits"), out['hits'], dtype=hits.dtype,
                      pix_type=pix_type)
        if coadd_fnames:
            savename_fnames = savename[:savename.find(".fits")] + ".txt"
            np.savetxt(savename_fnames.format("fnames"), sorted(out['filenames']), fmt='%s')

    if coadd_hits:
        return wmap, weights, out['hits']
    else:
        return wmap, weights


def make_full(template, split_pair, nbundles, pix_type, coadd_hits=True,
              coadd_fnames=False, savename=None, return_maps=True, **read_map_kwargs):
    """Add two splits to make a 'full' split for each bundle.

    Parameters
    ----------
    template: str
        Formattable string filename with args split, ibundle, map_type
    split_pair: 2-tuple
        Pair of string split identifiers to be summed
    nbundles: int
        Number of bundles
    pix_type: str
        hp or car
    coadd_hits: bool
        If True also add/return hits
    coadd_fnames: bool
        If True also add fnames
    savename: str
        Formattable string filename with args ibundle, map_type
    return_maps: bool
        If True return a list of (map, weights, hits) for all bundles.
        Can be set to False to save memory if many bundles are used.
    """
    out = []
    for ibundle in range(nbundles):
        sn = savename.format(ibundle, "{}") if savename is not None else None
        ans = coadd_bundles(template.format("{}", ibundle, "{}"),
                            split_pair, pix_type, coadd_hits=coadd_hits,
                            coadd_fnames=coadd_fnames, savename=sn, **read_map_kwargs)
        if return_maps:
            out.append(ans)
    if return_maps:
        return out

##############################################################################
## Misc Utility ##
##############################################################################
def filter_by_atomic_list(arr, atomic_list, obs_id_only=False, return_index=False):
    """ Filter an array by atomic_list

    Parameters
    ----------
    arr: Array
         First column should be obs_ids. Next columns should be wafer, freq_channel, if also filtering by them.
         May also be a DataFrame; see filter_by_atomic_list_df
    atomic_list: Array
        Array of [(obs_id, wafer, freq_channel), ...]
    obs_id_only: bool
        If True filter with only obs_id. Else also use wafer, freq_channel.
    return_index: bool
        If True also return indices of selected rows.

    Returns
    -------
    arr: Array
         Filtered array
    ind: Array [optional]
         Indices of selected rows
    """
    if type(arr) is pd.DataFrame:
        return filter_by_atomic_list_df(arr, atomic_list, obs_id_only, return_index)
    if atomic_list is None:
        if return_index:
            return arr, slice(None)
        else:
            return arr
    else:
        atomic_list = np.asarray(atomic_list)
        if obs_id_only:
            arr_query = arr if arr.ndim == 1 else arr[:,0]
            atomic_list_query = atomic_list if atomic_list.ndim == 1 else atomic_list[:,0]
            ind = np.isin(arr_query, atomic_list_query)
        else:
            tags1 = [' '.join(line[:3]) for line in arr]
            tags2 = [' '.join(line) for line in atomic_list]
            ind = np.isin(tags1, tags2)
        if return_index:
            return arr[ind], ind
        else:
            return arr[ind]

def filter_by_atomic_list_df(df, atomic_list, obs_id_only=False, return_index=False, reindex=True):
    """Filter a DataFrame by atomic_list; see filter_by_atomic_list"""
    if 'wafer' in df.columns and 'freq_channel' in df.columns:
        arr = np.array([df.obs_id, df.wafer, df.freq_channel]).T
    else:
        arr = df.obs_id.to_numpy()
    _, ind = filter_by_atomic_list(arr, atomic_list, obs_id_only, return_index=True)
    if reindex:
        return (df[ind]).reset_index(drop=True)
    else:
        return df[ind]

def extract_ws_freq(input_str):
    """
    Extract 'ws' and 'freq' from the input string.

    Parameters
    ----------
    input_str: str
        Input string containing 'ws' and 'freq' information.

    Returns
    -------
    ws: str
        Extracted 'ws' value.
    freq: str
        Extracted 'freq' value.
    """
    pattern = r'ws\d+|f\d+'
    matches = re.findall(pattern, input_str)
    ws = next((match for match in matches if match.startswith('ws')), None)
    freq = next((match for match in matches if match.startswith('f')),
                None)
    return ws, freq

def get_basename(prefix_path, depth):
    """ Get the base filename from a path, to a level given by depth.
    depth=0 is just the filename, depth=1 includes the first directory, etc.
    """
    split_prefix = [dr for dr in prefix_path.split('/') if dr]  # Split and remove double slashes
    return '/'.join(split_prefix[-depth-1:])

def get_abscal(abscal_dict, wafers, freqs):
    """ Get an array of abscal values for specific wafers/freqs.

    Parameters
    ----------
    abscal_dict: dict
        Nested dict in format {'ws0': {'f090': 1, 'f150': 1}, ...}.
        Maps will be multiplied by the abscal factor.
    wafers: list
        List of wafer identifiers as given in abscal_dict
    freqs: list
        List of freq identifiers as given in abscal_dict

    Returns
    -------
    abscal: numpy.array
        Array of multiplicative abscal factors with the same shape as wafers/freqs
    """
    if abscal_dict:
        abscal = np.array([abscal_dict[wafer][freq] for wafer, freq in zip(wafers, freqs)])
    else:
        abscal = np.ones(len(wafers))
    return abscal

def validate_map_string_format(map_string_format, wafer_tag, patch_tag):
    """Check that map filename format string has required and requested optional format tags."""
    for required_tag in ["{split}", "{bundle_id}", "{freq_channel}", "{map_type}"]:
        if required_tag not in map_string_format:
            raise ValueError(f"map_string_format does not have \
                               required placeholder {required_tag}")
    for optional_tag, tag_val in zip(["{wafer}", "{patch}"], [wafer_tag, patch_tag]):
        if optional_tag not in map_string_format and tag_val:
            print(f"Warning: map_string_format does not have optional \
                   placeholder {optional_tag} but value is passed")

def get_split_tag(split_intra_obs, split_inter_obs, coadd_pair, full_name='full'):
    """Parse intra/inter obs and set correct split tag."""
    # Map naming convention
    if (split_intra_obs, split_inter_obs) == (None, None):
        split_tag = full_name
    elif split_inter_obs is not None:
        # Inter; potentially with summed intras
        split_tag = split_inter_obs
    elif split_intra_obs is not None:
        if list(split_intra_obs) == list(coadd_pair):
            split_tag = full_name
        else:
            split_tag = split_intra_obs
    if isinstance(split_tag, list):
        split_tag = '_'.join(split_tag)
    return split_tag

def add_patch_to_query_restrict(patch, query_restrict=""):
    """Take a string patch name and add the correct az query to an existing restrict."""
    if patch is None:
        return query_restrict

    if patch == "south":
        patch_query = "(azimuth > 90 AND azimuth < 270)"
    elif patch == "north":
        patch_query = "(azimuth < 90 OR azimuth > 270)"
    else:
        raise ValueError(f"patch {patch} not recognized.")

    if patch_query in query_restrict:
        return query_restrict # Don't duplicate

    if query_restrict:
        query_restrict += " AND "
    query_restrict += patch_query
    return query_restrict

def load_atomic_list(atomic_list_fn):
    """Load an atomic list from .npy or .npz format."""
    if atomic_list_fn is None:
        return None
    if '.npz' in atomic_list_fn:
        atomic_list = np.load(atomic_list_fn)["atomic_list"]
    else:
        atomic_list = np.load(atomic_list_fn)
    return atomic_list

##############################################################################
## SQLite ##
##############################################################################
def _dbquery(db, query):
    cursor = db.cursor()
    result = cursor.execute(query).fetchall()
    return np.asarray(result).flatten()

##############################################################################
## Internal Functions ##
##############################################################################
def _check_pix_type(pix_type):
    """
    Error handling for pixellization types.

    Parameters
    ----------
    pix_type : str
        Pixellization type. Admissible values are "hp", "car.
    """
    if not (pix_type in ['hp', 'car']):
        raise ValueError(f"Unknown pixelisation type {pix_type}.")

def _get_map_template_car(template_map=None, res=5., dec_cut=None,
                          variant='fejer1', dtype=np.float64):
    """
    Get a map template for CAR

    Parameters
    ----------
    template_map : str
        File name for map or geometry file to use as template
    res : int
        Resolution of the map in arcmin if template_map is None
    dec_cut : tuple
        Min/max dec in deg if template_map is None
    variant : str
        CAR variant to use if template_map is None. 'fejer1' or 'cc'.
    dtype : np.dtype
        Data type.
    """
    if template_map is not None:
        if isinstance(template_map, str):
            shape, wcs = enmap.read_map_geometry(template_map)
        else:  # Assume we were passed a pre-loaded map
            shape, wcs = template_map.geometry
        shape = shape[-2:]
    elif dec_cut is not None:
        print(f"Using band geometry with dec_cut = {dec_cut}")
        shape, wcs = enmap.band_geometry(
            (np.deg2rad(dec_cut[0]), np.deg2rad(dec_cut[1])),
            res=np.deg2rad(res/60), variant=variant
        )
    else:
        print("Using full-sky geometry.")
        shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60), proj='car',
                                            variant="fejer1")

    atom_coadd = enmap.zeros((3, *shape), wcs, dtype=dtype)
    return atom_coadd

def _get_map_template_hp(template_map=None, nside=512, dtype=np.float64):
    """
    Get a map template for healpix

    Parameters
    ----------
    template_map : str
        File name for map or geometry file to use as template
    nside : int
        NSIDE to use if template_map is None
    dtype : np.dtype
        Data type.
    """
    if template_map is not None:
        if isinstance(template_map, str):
            temp = hp.read_map(template_map)
        else:
            temp = template_map  # Assume we were passed a pre-loaded map
        npix = temp.shape[-1]
    else:
        npix = 12 * nside**2
    atom_coadd = np.zeros((3, npix), dtype=dtype)
    return atom_coadd

def _add_map(imap, omap, pix_type):
    """Add a single map imap to an existing omap. omap is modified in place."""
    _check_pix_type(pix_type)
    if pix_type == 'hp':
        omap += imap
    elif pix_type == 'car':
        enmap.extract(imap, omap.shape, omap.wcs, omap=omap,
                      op=np.ndarray.__iadd__)

def _make_parallel_proc(fn, parallelizor):
    """Parallelize a coaddition function using ProcessPoolExecutor.

    Parameters
    ----------
    fn: function
        coaddition function with input params (list of filenames, template map)
        and return: coadded map
    parallelizor: tuple
        (MPICommExecutor or ProcessPoolExecutor, as_completed_callable, num_workers)

    Returns
    -------
    fn: function
        Parallelized coaddition function with same input params, plus optional
        nproc=num_workers from parallelizor
    """
    exe, as_completed, nproc = parallelizor
    def parallel_fn(filenames, template, *args, nproc=nproc, **kwargs):
        ibin = int(np.ceil(len(filenames)/nproc))
        slices = [slice(iproc*ibin, (iproc+1)*ibin) for iproc in range(nproc)]
        out = None

        futures = [exe.submit(fn, filenames, template, *args,
                              islice=slices[iproc], **kwargs)
                   for iproc in range(nproc)]
        for future in as_completed(futures):
            if out is None:
                out = future.result()
            else:
                out += future.result()
            futures.remove(future)

        return out
    return parallel_fn

##############################################################################
## Config ##
##############################################################################
@dataclass
class Cfg:
    """
    Class to configure bundling

    Args
    --------
    bundle_db: str
        Path to bundling database
    atomic_db: str
        Path to atomic map database
    n_bundles: int
        Number of map bundles
    seed: int
        Random seed that determines the composition of bundles
    query_restrict: str
        SQL query to restrict obs from the atomic database
    only_make_db: bool
        Only make bundling database and do not bundle maps
    patch: str
        'north', 'south', or None. May be a list of strings.
    inter_obs_props: dict
        Null properties for bundling database.
        Keys should be strings of (inter-obs null test)
        props available in atomic db.
        Values can be:
          - "median" to separate into two groups based on median values
          - {"splits": val_splits, "names": [name1, name2, ...]}
          - None to use each string value in the atomic db as its own group
        val_splits can be:
          - [(min1, max1), (min2, max2), ...] to pick vals in a numerical range
          - [(str1, str2, ...), (str3, str4, ...)] to group string values
    overwrite: bool
        Overwrite database if it exists
    pix_type: str
        'hp' or 'car'
    map_dir: str
        Path to directory containing atomic maps
    output_dir: str
        Path to output directory
    map_string_format: str
        String formatting for output bundles;
        must contain {name_tag} and {bundle_id}.
    freq_channel: str
        Frequency channel, e.g. 'f090'. May be a list of strings.
    intra_obs_splits: list
        List of split labels for intra-obs splits, e.g. 'scan_left'.
    intra_obs_pair: list
        Pair of intra-obs labels that will be added to make full obs
        for inter-obs splits
    inter_obs_splits:
        List of inter-obs split names for which to create bundles
    car_map_template: str
        Path to CAR map or geometry to be used as template
    wafer: str
        Wafer label, e.g. 'ws0'. May be a list of strings.
    save_fnames: bool
        Save the atomic map filenames for each bundle
    atomic_list: str
        Path to npy file of atomic map names to restrict the atomic db
    abscal: dict
        Multiplicative abscals {'ws0': {'f090': 0.8, 'f150': 0.9}, ...}
    coadd_splits_name: str
        "split" name for the coadd of two splits
    coadd_split_pair: list
        List of two (or more) splits to coadd
    coadd_bundles_splitname: str
        Split name for which to coadd all bundles to a full map.
        May also be a list of names.
    n_sims: int
        Number of sign flip realisations
    make_plots: bool
        If True make and save plots of the bundles
    tel: str
        **Deprecated** This doesn't do anything but kept for config compatibility
    nproc: int
        **Deprecated** This doesn't do anything but kept for config compatibility
    """
    bundle_db: str
    atomic_db: str
    n_bundles: int
    seed: int = 0
    query_restrict: str = ""
    only_make_db: bool = False
    patch: Optional[str] = None
    inter_obs_props: Optional[dict] = None
    overwrite: bool = False
    pix_type: str = "hp"
    map_dir: Optional[str] = None
    output_dir: Optional[str] = None
    map_string_format: Optional[str] = None
    freq_channel: Optional[str] = None
    intra_obs_splits: Optional[list] = None
    intra_obs_pair: Optional[list] = None
    inter_obs_splits: Optional[list] = None
    car_map_template: Optional[str] = None
    wafer: Optional[str] = None
    save_fnames: bool = False
    atomic_list: Optional[str] = None
    abscal: Optional[dict] = None
    coadd_splits_name: str = "full"
    coadd_split_pair: Optional[list] = None
    coadd_bundles_splitname: Optional[str] = None
    n_sims: Optional[int] = None
    make_plots: bool = False
    tel: Optional[str] = None
    nproc: Optional[int] = None

    def __post_init__(self):
        # Process patch argument
        if type(self.patch) is str or self.patch is None:
            self.patch_list = [self.patch]
        else:
            self.patch_list = self.patch
            self.patch = None

        self._update_attributes()

    def _update_attributes(self):
        """Do internal updating of certain attributes"""
        # Check valid pixelization
        _check_pix_type(self.pix_type)
        # Load the atomic list
        if type(self.atomic_list) is str:
            self.atomic_list = load_atomic_list(self.atomic_list)
        # Update query restrict with patch
        self.query_restrict_patch = add_patch_to_query_restrict(self.patch, query_restrict=self.query_restrict)

        # Update bundle db
        patch_tag = "" if self.patch is None else self.patch
        bundle_db_full = [(bundle_db.format(patch=patch_tag, seed=self.seed)).replace("__", "_") for bundle_db in np.atleast_1d(self.bundle_db)]
        self.bundle_db_full = bundle_db_full[0] if (type(self.bundle_db) is str) else bundle_db_full

    def update(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self._update_attributes()

    @classmethod
    def from_yaml(cls, path) -> "Cfg":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
            return cls(**d)

    def copy(self):
        return deepcopy(self)

def child_config(config, **kwargs):
    """Add key-value pairs in **kwargs to a copied config object and return."""
    config1 = config.copy()
    config1.update(**kwargs)
    return config1
