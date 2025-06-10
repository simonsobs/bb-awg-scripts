from pixell import enmap, enplot

#fname = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/transfer_function/butter4_cutoff_1e-2/atomic_sims/0000/pureE_20.0arcmin_fwhm30.0_sim0000_CAR_obs_1722069113_satp1_1111111_ws0_f090_scan_left_wmap.fits"
#out_fname = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/transfer_function/butter4_cutoff_1e-2/plots/pureE_20.0arcmin_fwhm30.0_sim0000_CAR_obs_1722069113_satp1_1111111_ws0_f090_scan_left_wmap"
fname = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/input_sims/pureE_20.0arcmin_fwhm30.0_sim0000_CAR.fits"
out_fname = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/input_sims/pureE_20.0arcmin_fwhm30.0_sim0000_CAR"

map = enmap.read_map(fname)

enplot.write(out_fname, enplot.plot(map))
