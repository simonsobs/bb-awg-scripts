import numpy as np
from pixell import enmap
import healpy as hp
from astropy.io import fits
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import Optional
from dataclasses import dataclass
import yaml


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


def coadd_maps(maps_list, weights_list, hits_list=None, sign_list=None,
               pix_type="hp", res_car=5., car_template_map=None,
               dec_cut_car=None, fields_hp=None, abscal=1, nproc=1):
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
    nproc: int
        Number of parallel processes to use. 1 for serial.

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
    sum_fn = _make_parallel_proc(sum_maps, nproc) if nproc > 1 else sum_maps

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


# The following functions are utility function inherited from
# sat-mapbundle-lib and elsewhere, but are not used (yet).
def _dbquery(db, query):
    cursor = db.cursor()
    result = cursor.execute(query).fetchall()
    return np.asarray(result).flatten()


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


def gen_masks_of_given_atomic_map_list_for_bundles(nmaps, nbundles):
    """
    Makes a list (length nbundles) of boolean lists (length nmaps)
    corresponding to the atomic maps that have to be coadded to make up a
    given bundle. This is done by uniformly distributing atomic maps into each
    bundle and, if necessary, looping through the bundles until the remainders
    have gone.

    Parameters
    ----------
    nmaps: int
        Number of atomic maps to distribute.
    nbundles: int
        Number of map bundles to be generated.

    Returns
    -------
    boolean_mask_list: list of list of str
        List of lists of booleans indicating the atomics to be coadded for
        each bundle.
    """

    n_per_bundle = nmaps // nbundles
    nremainder = nmaps % nbundles
    boolean_mask_list = []

    for idx in range(nbundles):
        if idx < nremainder:
            _n_per_bundle = n_per_bundle + 1
        else:
            _n_per_bundle = n_per_bundle

        i_begin = idx * _n_per_bundle
        i_end = (idx+1) * _n_per_bundle

        _m = np.zeros(nmaps, dtype=np.bool_)
        _m[i_begin:i_end] = True

        boolean_mask_list.append(_m)

    return boolean_mask_list


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


def _add_map(imap, omap, pix_type):
    """Add a single map imap to an existing omap. omap is modified in place."""
    _check_pix_type(pix_type)
    if pix_type == 'hp':
        omap += imap
    elif pix_type == 'car':
        enmap.extract(imap, omap.shape, omap.wcs, omap=omap,
                      op=np.ndarray.__iadd__)


def _make_parallel_proc(fn, nproc_default):
    """Parallelize a coaddition function using ProcessPoolExecutor.

    Parameters
    ----------
    fn: function
        coaddition function with input params (list of filenames, template map)
        and return: coadded map
    nproc_default: int
        Default number of processes to use

    Returns
    -------
    fn: function
        Parallelized coaddition function with same input params, plus optional
        nproc=nproc_default
    """
    def parallel_fn(filenames, template, *args, nproc=nproc_default, **kwargs):
        ibin = int(np.ceil(len(filenames)/nproc))
        slices = [slice(iproc*ibin, (iproc+1)*ibin) for iproc in range(nproc)]
        out = None
        with ProcessPoolExecutor(nproc) as exe:
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


def coadd_bundles(template, sum_vals, pix_type, do_hits=True, savename=None,
                  **read_map_kwargs):
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
    do_hits: bool
        If True also add/return hits
    savename: str
        Formattable string filename with args map_type
    read_map_kwargs: dict
        kwargs passed through to read_map
    """
    out = []
    for val in sum_vals:
        imap = read_map(template.format(val, 'map'), pix_type=pix_type,
                        **read_map_kwargs)
        weights = read_map(template.format(val, 'weights'), pix_type=pix_type,
                           **read_map_kwargs)
        if len(out) == 0:
            out.append(imap * weights)
            out.append(weights)
        else:
            out[0] += imap * weights
            out[1] += weights
        if do_hits:
            hits = read_map(template.format(val, 'hits'), pix_type=pix_type,
                            **read_map_kwargs)
            if len(out) < 3:
                out.append(hits)
            else:
                out[2] += hits
    del imap
    wmap, weights = out[:2]
    good = weights > 0
    wmap[good] /= weights[good]

    if savename is not None:
        write_map(savename.format("map"), wmap, dtype=wmap.dtype,
                  pix_type=pix_type)
        write_map(savename.format("weights"), weights, dtype=weights.dtype,
                  pix_type=pix_type)
        if do_hits:
            write_map(savename.format("hits"), out[2], dtype=hits.dtype,
                      pix_type=pix_type)
    if do_hits:
        return wmap, weights, out[2]
    else:
        return wmap, weights


def make_full(template, split_pair, nbundles, pix_type, do_hits=True,
              savename=None, return_maps=True, **read_map_kwargs):
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
    do_hits: bool
        If True also add/return hits
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
                            split_pair, pix_type, do_hits=do_hits,
                            savename=sn, **read_map_kwargs)
        if return_maps:
            out.append(ans)
    if return_maps:
        return out


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
    nproc: int
        Number of parallel processes to use in coadd
    atomic_list: str
        Path to npy file of atomic map names to restrict the atomic db
    abscal: bool
        Apply stored absolute calibration factors
    tel: str
        Telescope identifier for abscal
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
    nproc: int = 1
    atomic_list: Optional[str] = None
    abscal: bool = False
    tel: Optional[str] = None

    def __post_init__(self):
        # Add extra defaults for private args not expected in config file
        self.null_prop_val_inter_obs = None
        self.split_label_intra_obs = None

    @classmethod
    def from_yaml(cls, path) -> "Cfg":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
            return cls(**d)
