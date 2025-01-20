import numpy as np
from pixell import enmap
import healpy as hp
from astropy.io import fits
import h5py


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
        kwargs = {"field": fields_hp, } if fields_hp is not None else {}
        m = hp.read_map(map_file, **kwargs)
    else:
        # print(map_file)
        m = enmap.read_map(map_file, geometry=geometry)
        if is_weights:
            # Read only TT, QQ, UU weights
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


def _coadd_maps_car(maps_list, weights_list, hits_list=None, sign_list=None,
                    res=10, template_map=None, dec_cut=None):  # (-75, 25)
    """
    Coadd a list of atomics maps in CAR format. See function `coadd_maps` for
    documentation.
    """
    assert len(maps_list) == len(weights_list)

    if hits_list is not None:
        assert len(hits_list) == len(maps_list)
    if sign_list is None:
        sign_list = [1.]*len(maps_list)
    else:
        assert len(sign_list) == len(maps_list)

    if template_map is not None:
        w = enmap.read_map(template_map)
        _shape = w.shape[-2:]
        geometry = (_shape, w.wcs)
        template_geom = geometry
        print("geometry", geometry)
    elif dec_cut is not None:
        print("dec_cut", dec_cut)
        template_geom = enmap.band_geometry(
            (np.deg2rad(dec_cut[0]), np.deg2rad(dec_cut[1])),
            res=np.deg2rad(res/60)
        )
    else:
        raise ValueError("Either geometry or dec_cut required.")

    shape, wcs = template_geom
    atom_coadd = enmap.zeros((3, *template_geom[0]), template_geom[1])
    weight_coadd = enmap.zeros((3, *template_geom[0]), template_geom[1])

    if hits_list is not None:
        hits_coadd = enmap.zeros(shape, wcs=wcs)

    for i, (atom, weight) in enumerate(zip(maps_list, weights_list)):
        # atom_coadd = enmap.insert(atom_coadd, atom * float(sign_list[i]),
        #                           op=np.ndarray.__iadd__)
        # weight_coadd = enmap.insert(weight_coadd, weight,
        #                             op=np.ndarray.__iadd__)
        # if hits_list is not None:
        #     hits_coadd = enmap.insert(hits_coadd, np.squeeze(hits_list[i]),
        #                               op=np.ndarray.__iadd__)
        # m_ = enmap.read_map(atom) #* abfac
        m_ = atom  # * abfac
        mask = np.isfinite(m_)
        m_[~mask] = 0.0
        m_ = enmap.extract(m_, shape, wcs)

        # ivar = enmap.read_map(weight)
        ivar = weight
        mask = np.isfinite(ivar)
        ivar[~mask] = 0.0
        ivar = enmap.extract(ivar, shape, wcs)
        # ivar2 = ivar * abfac

        if hits_list is not None:
            # hits = enmap.read_map(hits_list[i])
            hits = hits_list[i]
            mask = np.isfinite(hits)
            hits[~mask] = 0.0
            hits = np.squeeze(enmap.extract(hits, shape, wcs))
            hits_coadd += hits

        atom_coadd += m_
        weight_coadd += ivar
        # weight_coadd_kcmb += ivar2

    # Cut zero-weight pixels
    weight_coadd[weight_coadd == 0] = np.inf
    atom_coadd /= weight_coadd

    if hits_list is not None:
        return atom_coadd, hits_coadd
    return atom_coadd


def _coadd_maps_hp(maps_list, weights_list, hits_list=None, sign_list=None):
    """
    Coadd a list of HEALPix atomics maps. See function `coadd_maps` for
    documentation.
    """
    assert len(maps_list) == len(weights_list)
    try:
        maps_list = np.array(maps_list, dtype=np.float32)
        weights_list = np.array(weights_list, dtype=np.float32)
    except:  # noqa
        raise ValueError("The input maps do not have the same size. "
                         "Are you sure you want to coadd healpix maps?")

    if hits_list is not None:
        assert len(hits_list) == len(maps_list)
    if sign_list is None:
        sign_list = [1.]*len(maps_list)
    else:
        assert len(sign_list) == len(maps_list)

    for i, (atom, weight) in enumerate(zip(maps_list, weights_list)):

        if i == 0:
            atom_coadd = np.zeros_like(atom)
            weight_coadd = np.zeros_like(weight)
            if hits_list is not None:
                hits_coadd = np.zeros_like(hits_list[i])

        atom_coadd += atom * float(sign_list[i])
        weight_coadd += weight
        if hits_list is not None:
            hits_coadd += hits_list[i]

    # Cut zero-weight pixels
    weight_coadd[weight_coadd == 0] = np.inf
    atom_coadd /= weight_coadd

    if hits_list is not None:
        return atom_coadd, hits_coadd
    return atom_coadd


def coadd_maps(maps_list, weights_list, hits_list=None, sign_list=None,
               pix_type="hp", res_car=10, car_template_map=None,
               dec_cut_car=None):
    """
    Coadd a list of weighted maps, a list of map weights, and
    (optionally) a list of hits maps corresponding to a set of atomics.
    Optionally, multiply weighted maps with a sign (+/-1) before coadding.

    Parameters
    ----------
    maps_list: list
        List of weighted TQU maps (numpy arrays).
    weights_list: list
        List of TQU map weights (numpy arrays).
    hits_list: list
        Optional; list of hits maps (numpy arrays). If None, ignore.
    sign_list:
        Optional; list of signs (+1 or -1) to multiply weighted maps with.
        If None, so not change maps at all.
    pix_type: str
        Pixelization assumed for input maps, either "hp" for HEALPix or
        "car" for CAR pixelization. Default: "hp"
    res_car: int
        Optional; pixel resolution of input CAR maps in arcminutes.
    dec_cut_car: tuple of int
        Optional; lower and upper declination cut in degrees, assumed
        for the inputCAR maps. Defaults to SAT-compatible sky region.

    Returns
    -------
    atom_coadd: np.array
        Coadded data TQU map, multiplied by the inverse coadded map weights.
    hits_coadd: np.array
        Optional; coadded hits map. Will only be returned if hits_list given.
    """
    if pix_type == "hp":
        return _coadd_maps_hp(
            maps_list, weights_list, hits_list=hits_list, sign_list=sign_list
        )
    elif pix_type == "car":
        return _coadd_maps_car(
            maps_list, weights_list, hits_list=hits_list, sign_list=sign_list,
            res=res_car, template_map=car_template_map, dec_cut=dec_cut_car
        )


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
