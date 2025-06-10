from pixell import enmap, enplot

fname = "/scratch/gpfs/SIMONSOBS/sat-iso/transfer_function/soopercool_inputs/planck_gal_mask_proj/planck_galmask_070_CAR_5arcmin.fits"
fname_out = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/soopercool_inputs/planck_gal_mask_proj/planck_galmask_070_CAR_20arcmin.fits"

map = enmap.read_map(fname)
omap = enmap.downgrade(map, 4)
enmap.write_map(fname_out, omap)
enplot.write(fname_out.replace(".fits", ""), enplot.plot(omap, min=-1, max=1))
