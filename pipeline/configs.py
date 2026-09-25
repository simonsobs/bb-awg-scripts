import numpy as np
import yaml
from typing import Union, Sequence, Optional
from dataclasses import dataclass
import sys
import copy
sys.path.append("bundling")
from bundling_utils import check_pix_type, load_atomic_list, add_patch_to_query_restrict

def yaml_loader(config):
    """
    Custom yaml loader to load the configuration file.
    """
    def path_constructor(loader, node):
        return "/".join(loader.construct_sequence(node))
    yaml.SafeLoader.add_constructor("!path", path_constructor)
    with open(config, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

@dataclass
class _Cfg():
    """Configuration file base class"""
    def copy(self):
        return copy.deepcopy(self)
    def _update_attributes(self):
        """Keep all attributes up-to-date when some are changed
        using update()"""
        return
    def update(self, **kwargs):
        """Add extra private args not expected in config file"""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._update_attributes()
    def __post_init__(self):
        self._update_attributes()

    @classmethod
    def from_yaml(cls, path) -> "Cfg":
        d = yaml_loader(path)
        sub_blocks = {'bundle_db_cfg': BundleDbCfg, 'bundling': BundleCfg, 'filtering': FilteringCfg, 'signflip': SignFlipCfg}
        for k, v in sub_blocks.items():
            if k in d.keys():
                d[k] = v(**d[k])
        return cls(**d)

    def child(self, **kwargs):
        """Add key-value pairs in **kwargs to a copied config object and return."""
        config1 = self.copy()
        for k, v in kwargs.items():
            if k not in vars(self):
                raise KeyError(f"{k} does not exist")
        config1.update(**kwargs)
        return config1

@dataclass
class BundleDbCfg(_Cfg):
    """
    Config class for making the Bundle DB.

    Args
    --------
    bundle_duration: int
        Width of the ctime bins that will be assigned bundles, in seconds.
        Can also be the string 'obs' to bundle by obs_id instead.
    bundle_t0: int
        ctime of the lowest bin for bundle assignment. Sets phase of the binning.
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
    atomic_list: str
        Path to npy file of atomic map names to restrict the atomic db
    only_make_db: bool
        Only make bundling database and do not bundle maps
    """
    bundle_duration: Union[int, str] = 86400
    bundle_t0: int = 1704121200
    inter_obs_props: Optional[dict] = None
    overwrite: bool = False
    atomic_list: Optional[str] = None
    only_make_db: bool = False

    def _update_attributes(self):
        # Load the atomic list
        if type(self.atomic_list) is str:
            self.atomic_list = load_atomic_list(self.atomic_list)

@dataclass
class BundleCfg(_Cfg):
    """
    Config class for making the Bundle DB.

    Args
    --------
    map_dir: str
        Path to directory containing atomic maps
    map_string_format: str
        String formatting for output bundles;
        must contain {name_tag} and {bundle_id}.
    output_dir_bundling: str
        Path to output directory
    wafer: str
        Wafer label, e.g. 'ws0'. May be a list of strings.
    abscal: dict
        Multiplicative abscals {'f090': {'ws0': 1, 'ws1': 1,...}, ...}
    make_plots: bool
        If True make and save plots of the bundles
    save_fnames: bool
        Save the atomic map filenames for each bundle
    coadd_splits_name: str
        "split" name for the coadd of two splits
    coadd_split_pair: list
        List of two (or more) splits to coadd
    coadd_bundles_splitname: str
        Split name for which to coadd all bundles to a full map.
        May also be a list of names.
    """
    map_dir: str
    map_string_format: str
    output_dir_bundling: Optional[str] = None
    wafer: Optional[str] = None
    abscal: Optional[dict] = None
    make_plots: bool = False
    save_fnames: bool = False
    coadd_splits_name: str = "full"
    coadd_split_pair: Optional[Sequence[str]] = None
    coadd_bundles_splitname: Optional[str] = None

@dataclass
class SignFlipCfg(_Cfg):
    """
    Config class for making the Bundle DB.

    Args
    --------
    output_dir_signflip: str
        Path to output directory
    n_sims: int
        Number of sign flip realisations
    overwrite_sf: bool
        If True overwrite existing signflip maps
    """
    output_dir_signflip: Optional[str] = None
    n_sims: Optional[int] = None
    overwrite_sf: bool = False

@dataclass
class FilteringCfg(_Cfg):
    """
    Class to configure filtering

    Args
    --------
    preprocess_config_init: str
        Path to preprocessing init yaml file
    preprocess_config_proc: str
        Path to preprocessing proc yaml file
    output_dir_filtering: str
        Path to output directory
    sim_dir: str
        Path to directory containing unfiltered input sims
    atomic_sim_dir: str
        Path to directory containing filtered atomic sims
    sim_string_format: str
        String formatting for unfiltered input sims
        must contain {sim_id} and {sim_type}.
    coadded_dirs: str
        Directory for coadded maps
    sim_ids: list
        Simulation seeds to be filtered, passed as integers
    sim_types: list
        Strings that define the simulation types to be filtered, e.g.
        ['pureT', 'pureE', 'pureB'], or ['cmbEB', 'cmbB'] etc.
    bundle_id: int
        Bundle ID to be filtered
    nside: int
        HEALPix NSIDE parameter
    fp_thin: int
        Focal plane thinning factor applied to the sim filtering
    nbatch_atomics: int
        Number of batches to divide the bundle into, based on random timestamp
        splits
    remove_atomics: bool
        Removes atomic maps from disk upon coadding them
    overwrite_atomics: bool
        Overwrites atomic sim maps if they exist
    verbosity: int
    """
    preprocess_config_init: str
    preprocess_config_proc: str
    output_dir_filtering: str
    sim_dir: str
    atomic_sim_dir: str
    sim_string_format: str
    coadded_dirs: Optional[str] = None
    sim_ids: Optional[list] = None
    sim_types: Optional[list] = None
    bundle_id: int = 0
    nside: Optional[int] = None
    fp_thin: Optional[int] = 8
    nbatch_atomics: Optional[int] = None
    remove_atomics: bool = False
    overwrite_atomics: bool = True
    verbosity: int = 2

@dataclass
class Cfg(_Cfg):
    """
    Class to configure bundling

    Args
    --------
    base_dir: str
        Optionally used at yaml level to reduce repetition in paths.
    bundle_db: str
        Path to bundling database
    patch: str
        'north', 'south', or None. May be a list of strings.
    n_bundles: int
        Number of map bundles
    seed: int
        Random seed that determines the composition of bundles
    atomic_db: str
        Path to atomic map database
    query_restrict: str
        SQL query to restrict obs from the atomic database
    freq_channel: str
        Frequency channel, e.g. 'f090'. May be a list of strings.
    pix_type: str
        'hp' or 'car'
    car_map_template: str
        Path to CAR map or geometry to be used as template
    intra_obs_splits: list
        List of split labels for intra-obs splits, e.g. 'scan_left'.
    intra_obs_pair: list
        Pair of intra-obs labels that will be added to make full obs
        for inter-obs splits
    inter_obs_splits:
        List of inter-obs split names for which to create bundles
    """
    # All
    base_dir: str
    bundle_db: str
    patch: Optional[Sequence[str]] = None
    # Bundle DB and Bundling
    n_bundles: Optional[int] = None
    seed: int = 0
    # Bundle DB and Filtering
    atomic_db: Optional[str] = None
    query_restrict: str = ""
    freq_channel: Union[str, Sequence[str], None] = None
    pix_type: str = 'car'
    car_map_template: Optional[str] = None
    intra_obs_splits: Optional[Sequence[str]] = None
    intra_obs_pair: Optional[Sequence[str]] = None
    inter_obs_splits: Optional[Sequence[str]] = None
    bundle_db_cfg: Optional[BundleDbCfg] = None
    bundling: Optional[BundleCfg] = None
    filtering: Optional[FilteringCfg] = None
    signflip: Optional[SignFlipCfg] = None

    @property
    def current_patch(self):
        patch_list = np.atleast_1d(self.patch)
        if patch_list.size > 1:
            raise ValueError("Operation undefined for multiple patches. Set 'patch' to a string or len-1 list.")
        return patch_list[0]
    @property
    def query_restrict_patch(self):
        return add_patch_to_query_restrict(self.current_patch, query_restrict=self.query_restrict)
    @property
    def bundle_db_full(self):
        patch_tag = "" if self.current_patch is None else self.current_patch
        bundle_db_full = [(bundle_db.format(patch=patch_tag, seed=self.seed)).replace("__", "_") for bundle_db in np.atleast_1d(self.bundle_db)]
        return bundle_db_full[0] if (type(self.bundle_db) is str) else bundle_db_full

    def _update_attributes(self):
        check_pix_type(self.pix_type)
