import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
import argparse


def main(args):
    fpairs = ["EE", "EB", "BE", "BB"]
    bins = np.load(args.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(bins["bin_low"], bins["bin_high"] + 1)
    lb = nmt_bins.get_effective_ells()
    lmax_plot = 600

    # Just insert labels and paths as needed
    tf_files = {args.tf_label: args.tf_file}

    colors = {args.tf_label: "navy"}
    linestyles = {args.tf_label: "-"}

    _, axes = plt.subplots(4, 4, figsize=(13, 13))
    mask = lb <= lmax_plot

    for lab in tf_files:
        tf = np.load(tf_files[lab])
        for i, f1 in enumerate(fpairs):
            for j, f2 in enumerate(fpairs):
                expected = 1. if f1 == f2 else 0.
                axes[i, j].axhline(expected, color="k", ls="--", zorder=6)

                if f1 != f2:
                    axes[i, j].set_ylim(-0.01, 0.01)
                else:
                    axes[i, j].set_ylim(0, 1.05)

                axes[i, j].errorbar(
                    lb[mask],
                    tf[f"{f1}_to_{f2}"][mask],
                    yerr=tf[f"{f1}_to_{f2}_std"][mask],
                    ls=linestyles[lab], lw=1.2,
                    #markerfacecolor="white",
                    #marker=".",
                    #markersize=6.2,
                    label=lab,
                    alpha=0.8,
                    color=colors[lab]
                )

                if f1 == "EE" and f2 == "BB":
                    axes[i, j].legend(fontsize=9, loc="lower right")

                axes[i, j].set_title(f"{f1} $\\rightarrow$ {f2}", fontsize=13)

                axes[i, j].set_xlim(2, 600)
                if i != 3:
                    axes[i, j].set_xticklabels([])

                if i == 3:
                    axes[i, j].set_xlabel(r"$\ell$", fontsize=13)
                if j == 0:
                    axes[i, j].set_ylabel(r"$T_{\ell}$", fontsize=13)

                axes[i, j].tick_params(axis='both', which='both', labelsize=12)

                if i != j:
                    axes[i, j].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.savefig(args.plot_fname, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot transfer function"
    )
    parser.add_argument("--plot_fname", help="Name for plot")
    parser.add_argument("--tf_label", help="Label for TF")
    parser.add_argument("--tf_file", help="File name for TF")

    args = parser.parse_args()

    main(args)
