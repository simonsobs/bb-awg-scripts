from pixell import enmap

temp = "/home/kw6905/bbdev/pwg-scripts/iso-sat-review/mapmaking/band_car_fejer1_5arcmin.fits"
geom = enmap.read_map_geometry(temp)

print("geom 1", geom)

#ogeom = enmap.downgrade_geometry(geom[0], geom[1], 4)
ogeom = enmap.upgrade_geometry(geom[0], geom[1], 5)


print("geom 2", ogeom)

enmap.write_map_geometry("band_car_fejer1_1arcmin.fits", ogeom[0], ogeom[1])
