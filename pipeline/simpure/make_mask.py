import sys
from pixell import enmap, enplot

sys.path.append("/home/kw6905/bbdev/SOOPERCOOL/soopercool")
import map_utils as mu

binary = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/transfer_function/satp1_iso_nsims5/f090/binary_mask_pureB_sim0000_SATp1_f090_south_science.fits"
apodized = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/transfer_function/satp1_iso_nsims5/f090/binary_apo10_C2.fits"
binary = enmap.read_map(binary)

analysis_mask = mu.apodize_mask(
        binary,
        apod_radius_deg=10.,
        apod_type="C2",
        pix_type="car"
    )

print(apodized.replace(".fits", ".png"))
enmap.write_map(apodized, analysis_mask)
enplot.write(apodized.replace(".fits", ""), enplot.plot(analysis_mask, min=-1, max=1))
