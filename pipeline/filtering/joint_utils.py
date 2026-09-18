import numpy as np

bundle_db_args = [bundle_db, atomic_db, n_bundles]
bundle_db_opt = {bundle_duration: 86400, bundle_t0: 1704121200, query_restrict: "", seed: 0,
                 inter_obs_props: None, overwrite: False, atomic_list: None, patch: None}
bundling_args = [bundle_db, n_bundles, pix_type, map_dir, freq_channel, map_string_format,
                 output_dir_bundling]
bundling_opt = {intra_obs_splits: None, intra_obs_pair: None, inter_obs_splits: None,
                car_map_template: None, wafer: None, save_fnames: False, abscal: None,
                coadd_splits_name: "full", coadd_split_pair: None, coadd_bundles_splitname: None,
                make_plots: False, only_make_db: False, base_dir: "", patch: None}
signflip_args = [bundle_db, n_bundles, pix_type, map_dir, freq_channel, map_string_format,
            output_dir_signflip, n_sims]
filering_args = [bundle_db, atomic_db, preprocess_config_init, preprocess_config_proc, sim_dir,
                 atomic_sim_dir, output_dir_filtering, sim_string_format, freq_channel, patch]
filtering_opt = {coadded_dirs: None, query_restrict: "", pix_type: "car", bundle_id: 0,
                 intra_obs_splits: None, sim_ids: None, sim_types, None, intra_obs_pair: None,
                 inter_obs_splits: None, car_map_template: None, nside: None, fp_thin: 8,
                 nbatch_atomics: None, remove_atomics: False, overwrite_atomics: True,
                 base_dir: None, verbosity: 2}

all_opt = list(bundle_db_opt + bundling_opt + filtering_opt)
allowed = bundle_db_args + bundling_args + filtering_args + signflip_args + all_opt

def check_inputs(required, **kwargs):
    input_args = np.array(list(kwargs))
    missing = required[~np.isin(required, input_args)]
    if missing.size > 0:
        raise ValueError(f"Required args {missing} missing")
    banned = input_args[~np.isin(input_args, np.concatenate([required, allowed]))]
    if banned.size > 0:
        raise ValueError(f"Unrecognized args {banned}")

class Cfg():
    def __init__(self, required, **kwargs):
        check_inputs(required, **kwargs)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def update(self, dict):
        # Add extra private args not expected in config file
        for k, v in dict.items():
            setattr(self, k, v)

    @classmethod
    def from_yaml(cls, path) -> "Cfg":
        d = yaml_loader(path)
        return cls(**d)
    
class BundleCfg(Cfg):
    def __init__(self, **kwargs):
        required = bundle_db_args + bundling_args
        return super().__init__(required)
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

    def update(self, dict):
        super().update(dict)
        self.update_attributes()

    
class SignflipCfg(BundleCfg):
    def __init__(self, **kwargs):
        return super().__init__(signflip_args)
class FilteringCfg(Cfg):
    def __init__(self, **kwargs):
        return super().__init__(filtering_args)

