import argparse
import re
import sys
import os
sys.path.append(
    "/home/kw6905/bbdev/SOOPERCOOL"
)
from soopercool import map_utils as mu
from soopercool import utils as su
import numpy as np


def main(args):
    """
    """
    do_plots = not args.no_plots
    verbose = args.verbose

    out_dir = args.out_dir

    masks_dir = f"{out_dir}/masks"
    plot_dir = f"{out_dir}/plots/masks"
    os.makedirs(masks_dir, exist_ok=True)
    if do_plots:
        os.makedirs(plot_dir, exist_ok=True)

    # If a global hits file is indicated in the paramter file, use it.
    if args.global_hits is not None:
        print("global hits")
        sum_hits = mu.read_map(
            args.global_hits,
            pix_type=args.pix_type,
            car_map_template=args.car_template
        )
        # Create binary
        binary = sum_hits.copy()
        binary[:] = 1
        binary[sum_hits == 0] = 0
    else:
        # Loop over the (map_set, id_bundles)
        # pairs to define a common binary mask
        maps = []
        n_bundles = args.n_bundles
        map_set = args.map_set
        map_dir = args.map_dir
        map_string_format = args.map_string_format

        for id_bundle in range(n_bundles):
            map_file = map_string_format.format(id_bundle=id_bundle)
            print(f"Reading map for {map_set} - bundle {id_bundle}")
            if verbose:
                print(f"    file_name: {map_dir}/{map_file}")

            map = mu.read_map(
                f"{map_dir}/{map_file}",
                pix_type=args.pix_type,
                car_map_template=args.car_template
            )
            maps.append(map)

        # Create binary and normalized hitmap
        binary = maps[0][1].copy()
        sum_maps = maps[0][1].copy()
        binary[:] = 1
        sum_maps[:] = 0
        for map in maps:
            binary[map[1] == 0] = 0
            sum_maps += map[1]
        sum_maps[binary == 0] = 0

    # Normalize and smooth hitmaps
    sum_maps = mu.smooth_map(sum_maps, fwhm_deg=1, pix_type=args.pix_type)
    sum_maps /= np.amax(sum_maps)

    # Save products
    mu.write_map(
        f"{masks_dir}/binary_mask.fits",
        binary,
        dtype=np.int32,
        pix_type=args.pix_type
    )
    mu.write_map(
        f"{masks_dir}/normalized_hits.fits",
        sum_maps,
        dtype=np.float32,
        pix_type=args.pix_type
    )

    if do_plots:
        mu.plot_map(
            binary,
            title="Binary mask",
            file_name=f"{plot_dir}/binary_mask",
            pix_type=args.pix_type,
            lims=[-1, 1]
        )

        mu.plot_map(
            sum_maps,
            title="Normalized hits",
            file_name=f"{plot_dir}/normalized_footprint",
            pix_type=args.pix_type,
            lims=[-1, 1]
        )

    analysis_mask = binary.copy()

    if args.galactic_mask is not None:
        print("Reading Galactic mask ...")
        if verbose:
            print(f"    file_name: {args.galactic_mask}")
        gal_mask = mu.read_map(args.galactic_mask,
                               pix_type=args.pix_type,
                               geometry=analysis_mask.geometry)
        if do_plots:
            mu.plot_map(
                gal_mask,
                title="Galactic mask",
                file_name=f"{plot_dir}/galactic_mask",
                pix_type=args.pix_type,
                lims=[-1, 1]
            )
        analysis_mask *= gal_mask

    if args.external_mask is not None:
        print("Reading external mask ...")
        if verbose:
            print(f"    file_name: {args.external_mask}")
        ext_mask = mu.read_map(args.external_mask,
                               pix_type=args.pix_type,
                               geometry=analysis_mask.geometry)
        if do_plots:
            mu.plot_map(
                ext_mask,
                title="External mask",
                file_name=f"{plot_dir}/external_mask",
                pix_type=args.pix_type,
                lims=[-1, 1]
            )
        analysis_mask *= ext_mask

    analysis_mask = mu.apodize_mask(
        analysis_mask,
        apod_radius_deg=args.apod_radius,
        apod_type=args.apod_type,
        pix_type=args.pix_type
    )

    if args.point_source_mask is not None:
        print("Reading point source mask ...")
        if verbose:
            print(f"    file_name: {args.point_source_mask}")
        ps_mask = mu.read_map(args.point_source_mask,
                              pix_type=args.pix_type,
                              geometry=analysis_mask.geometry)
        ps_mask = mu.apodize_mask(
            ps_mask,
            apod_radius_deg=args.apod_radius_point_source,
            apod_type=args.apod_type,
            pix_type=args.pix_type
        )
        if do_plots:
            mu.plot_map(
                ps_mask,
                title="Point source mask",
                file_name=f"{plot_dir}/point_source_mask",
                pix_type=args.pix_type,
                lims=[-1, 1]
            )

        analysis_mask *= ps_mask

    # Weight with hitmap
    analysis_mask *= sum_maps
    mu.write_map(
        f"{masks_dir}/analysis_mask.fits",
        analysis_mask,
        pix_type=args.pix_type
    )

    if do_plots:
        mu.plot_map(
            analysis_mask,
            title="Analysis mask",
            file_name=f"{plot_dir}/analysis_mask_apo{apod_radius}_{apod_type}",
            pix_type=args.pix_type,
            lims=[-1, 1]
        )

    # Compute and plot spin derivatives
    if args.pix_type == "car":
        print("WARNING: Spin derivatives are not implemented yet. SKIPPING.")
    else:
        first, second = su.get_spin_derivatives(analysis_mask)

        if do_plots:
            mu.plot_map(
                first,
                title="First spin derivative",
                file_name=f"{plot_dir}/first_spin_derivative"
            )
            mu.plot_map(
                second,
                title="Second spin derivative",
                file_name=f"{plot_dir}/second_spin_derivative"
            )

        if args.verbose:
            print("---------------------------------------------------------")
            print("Using custom mask. "
                  "Its spin derivatives have global min and max of:")
            print("first:     ", np.amin(first), np.amax(first),
                  "\nsecond:    ", np.amin(second), np.amax(second))
            print("---------------------------------------------------------")

    print("\nSUMMARY")
    print("-------")
    print(f"Wrote analysis mask to {masks_dir}/analysis_mask.fits")
    print("Parameters")
    print(f"    Galactic mask: {args.galactic_mask}")
    print(f"    External mask: {args.external_mask}")
    print(f"    Point source mask: {args.point_source_mask}")
    print(f"    Apodization type: {args.apod_type}")
    print(f"    Apodization radius: {args.apod_radius}")
    print(f"    Apodization radius point source: {args.apod_radius_point_source}") # noqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get analysis mask")
    parser.add_argument("--out_dir", help="Output directory")
    parser.add_argument("--global_hits", help="Global hits map", default=None)
    parser.add_argument("--pix_type", help="Pixel type: car or hp")
    parser.add_argument("--car_template", help="Car geometry template")
    parser.add_argument("--n_bundles", help="Number of bundles", type=int)
    parser.add_argument("--map_set", help="Name of the map set")
    parser.add_argument("--map_dir", help="Map directory")
    parser.add_argument("--map_string_format", help="String format of the map file")
    parser.add_argument("--galactic_mask", help="Path to Galactic mask",
                        default=None)
    parser.add_argument("--external_mask",
                        help="External mask to multiply with",
                        default=None)
    parser.add_argument("--point_source_mask",
                        help="Point source mask", default=None)
    parser.add_argument("--apod_radius", help="Apodization radius in degreed",
                        default=10)
    parser.add_argument("--apod_type", help="Apodization type; C1 or C2",
                        default="C1")
    parser.add_argument("--apod_radius_point_source",
                        help="Apodization radius for point sources in degrees",
                        default=1)

    parser.add_argument("--verbose", help="Verbose mode",
                        action="store_true")
    parser.add_argument("--no-plots", help="Plot the results",
                        action="store_true")

    args = parser.parse_args()

    main(args)
