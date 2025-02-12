import os
import bundling_utils
import argparse

def main(args):
    template = os.path.join(args.output_dir,
                            args.map_string_format.format(name_tag=args.freq_channel+"_{}", bundle_id="{}"))
    template = template.replace("map.fits", "{}.fits")

    if args.make_full:
        print("Making full maps")
        savename = template.format("full", "{}", "{}")
        bundling_utils.make_full(template,
                                 args.split_pair,
                                 args.n_bundles,
                                 args.pix_type,
                                 do_hits=True,
                                 savename=savename,
                                 return_maps=False)

    if args.add_bundles:
        print("Co-adding  bundles")
        template = template.format("full", "{}", "{}")
        sum_vals = list(range(args.n_bundles))
        savename = template.format("!", "{}").replace("_bundle!", "")
        bundling_utils.coadd_bundles(template,
                                     sum_vals,
                                     args.pix_type,
                                     do_hits=True,
                                     savename=savename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coadd bundled maps")

    parser.add_argument(
        "--freq_channel",
        help="Frequency channel, e.g. 'f090'"
    )
    parser.add_argument(
        "--n_bundles",
        help="Number of map bundles.",
        type=int,
        required=True
    )
    parser.add_argument(
        "--split_pair",
        nargs="*",
        help="Split labels to add to make full coadd",
        default=None
    )
    parser.add_argument(
        "--pix_type",
        help="Pixel type, either 'hp' or 'car'.",
        default="hp"
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory."
    )
    parser.add_argument(
        "--map_string_format",
        help="String formatting; must contain {name_tag} and {bundle_id}."
    )

    parser.add_argument(
        "--make_full",
        action="store_true",
        help="Coadd splits to make a full map"
    )
    parser.add_argument(
        "--add_bundles",
        action="store_true",
        help="Coadd bundles to make a full coadd"
    )

    args = parser.parse_args()

    main(args)
