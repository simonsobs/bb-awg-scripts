import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
from itertools import product

base_dir = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/transfer_function"
plot_dir = "/home/kw6905/bbdev/bb-awg-scripts/pipeline/simpure/satp3/plots"
binning_file = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/soopercool_inputs/binning/binning_car_lmax540_deltal10_large_first_bin.npz"  # noqa
which_tf_panels = "EE_to_BB"
normalize_to_cmb_pol = False
lmax_plot = 600

tf_files = {
    "butter4_nmt_pure": "butter4_cutoff_1e-2/soopercool_outputs_nsims_200_nmt_purify/transfer_functions/transfer_function_SATp3_f090_south_science_x_SATp3_f090_south_science.npz",  # noqa
    "butter4_nopure": "butter4_cutoff_1e-2/soopercool_outputs_nsims_200/transfer_functions/transfer_function_SATp3_f090_south_science_x_SATp3_f090_south_science.npz",  # noqa
}


def get_theory_cls(cosmo_params, lmax, lmin=0):
    """
    """
    import camb
    params = camb.set_params(**cosmo_params)
    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit='K', raw_cl=True)
    lth = np.arange(lmin, lmax+1)

    cl_th = {
        "TT": powers["total"][:, 0][lmin:lmax+1],
        "EE": powers["total"][:, 1][lmin:lmax+1],
        "TE": powers["total"][:, 3][lmin:lmax+1],
        "ET": powers["total"][:, 3][lmin:lmax+1],
        "BB": powers["total"][:, 2][lmin:lmax+1]
    }
    for spec in ["EB", "TB"]:
        cl_th[spec] = np.zeros_like(lth)

    return lth, cl_th


def get_planck_cosmo_params():
    """
    """
    cosmo = {
       "cosmomc_theta": 0.0104085,
       "As": 2.1e-9,
       "ombh2": 0.02237,
       "omch2": 0.1200,
       "ns": 0.9649,
       "Alens": 1.0,
       "tau": 0.0544,
       "r": 0.0,
    }
    return cosmo


field_pairs = ["TT", "TE", "TB", "ET", "EE", "EB", "BT", "BE", "BB"]
if which_tf_panels == "all":
    _, axes = plt.subplots(9, 9, figsize=(28, 28))
    def tf_panels_iterator():
        for i, f1 in enumerate(field_pairs):
            for j, f2 in enumerate(field_pairs):
                yield i, f1, j, f2
    text_fontsize = 8
elif which_tf_panels == "all_BB":
    _, axes = plt.subplots(9, 2, figsize=(7, 28))
    def tf_panels_iterator():
        for i, f1 in enumerate(field_pairs):
            yield i, f1, 0, "BB"
            yield i, "BB", 1, f1
    text_fontsize = 10
elif which_tf_panels == "all_EE":
    _, axes = plt.subplots(9, 2, figsize=(7, 28))
    def tf_panels_iterator():
        for i, f1 in enumerate(field_pairs):
            yield i, f1, 0, "EE"
            yield i, "EE", 1, f1
    text_fontsize = 10
elif which_tf_panels in [f"{fp1}_to_{fp2}"
                         for fp1, fp2 in product(field_pairs, field_pairs)]:
    from matplotlib.ticker import AutoMinorLocator

    _, axes = plt.subplots(1, 1, figsize=(8, 6), squeeze=False)
    fp1, fp2 = which_tf_panels.split("_to_")
    axes[0, 0].xaxis.set_ticks(np.arange(0, lmax_plot, 50))
    minor_locator = AutoMinorLocator(2)
    axes[0, 0].xaxis.set_minor_locator(minor_locator)
    axes[0, 0].tick_params(
        axis='x', which="both",
        bottom=True, top=True, labeltop=False,
        direction="in"
    )
    text_fontsize = 12

    def tf_panels_iterator():
        yield 0, fp1, 0, fp2


bins = np.load(binning_file)
nmt_bins = nmt.NmtBin.from_edges(bins["bin_low"], bins["bin_high"] + 1)
lmax_bins = nmt_bins.get_ell_max(nmt_bins.get_n_bands() - 1)
lb = nmt_bins.get_effective_ells()
mask = lb <= lmax_plot
colors = plt.cm.tab20(np.linspace(0, 1, 13))


if normalize_to_cmb_pol:
    _, clth = get_theory_cls(get_planck_cosmo_params(), lmax=lmax_bins)
    clb_theory = {fp: nmt_bins.bin_cell(cl)[mask] for fp, cl in clth.items()}

    norm_factor = {
        (fp1, fp2): (clb_theory[fp1] / clb_theory[fp2], fp2)
        for fp1 in ["TT", "TE", "ET", "EE", "BB"]
        for fp2 in ["TT", "TE", "ET", "EE", "BB"]
    }
    # plt.loglog(lb, clb_theory["EE"], "r")
    # plt.loglog(range(lmax + 1), clth["EE"], "k--")
    # plt.loglog(lb, clb_theory["BB"], "b")
    # plt.loglog(range(lmax + 1), clth["BB"], "c--")
    # plt.savefig(f"{plot_dir}/test_camb.png")


for ilab, (tf_label, tf_file) in enumerate(tf_files.items()):
    tf = np.load(f"{base_dir}/{tf_file}")
    for i, f1, j, f2 in tf_panels_iterator():
        transfer = tf[f"{f1}_to_{f2}"][mask]
        expected = 1. if f1 == f2 else 0.
        axes[i, j].axhline(expected, color="k", ls="--", zorder=6)

        if normalize_to_cmb_pol and (f1, f2) in norm_factor:
            transfer *= norm_factor[f1, f2][0]
            axes[i, j].text(0.35, 0.95,
                            f"units of Planck {norm_factor[f1, f2][1]}",
                            transform=axes[i,j].transAxes,
                            fontsize=text_fontsize,
                            va='top', ha='left')
            if (f1, f2) == ("EE", "BB"):
                axes[i, j].set_ylim(-.5, .5)

        if not normalize_to_cmb_pol and f1 != f2 and which_tf_panels == "all":
            axes[i, j].set_ylim(-0.01, 0.01)

        if f1 == f2:
            axes[i, j].set_ylim(0, 1.05)

        axes[i, j].plot(
            lb[mask],
            transfer,
            ls="-",
            lw=1.2,
            label=tf_label,
            alpha=0.8,
            color=colors[ilab]
        )

        if j == len(axes[0, :]) - 1 and i == 0:
            axes[i, j].legend(fontsize=9, loc="upper left",
                              bbox_to_anchor=(1.1, 1.0))

        axes[i, j].set_title(f"{f1} $\\rightarrow$ {f2}", fontsize=13)

        axes[i, j].set_xlim(2, lmax_plot)
        if i != len(axes[:, j]) - 1:
            axes[i, j].set_xticklabels([])

        if i == len(axes[:, j]) - 1:
            axes[i, j].set_xlabel(r"$\ell$", fontsize=13)
        if j == 0:
            axes[i, j].set_ylabel(r"$T_{\ell}$", fontsize=13)

        axes[i, j].tick_params(axis='both', which='both', labelsize=12)

        if i != j:
            try:
                axes[i, j].ticklabel_format(axis="y", style="sci",
                                            scilimits=(0, 0))
            except:  # noqa
                pass

    norm_lab = "_normalized" if normalize_to_cmb_pol else ""
    plot_fname = f"{plot_dir}/transfer_functions_{which_tf_panels}{norm_lab}.pdf"  # noqa
    print("Saved to", plot_fname)
    plt.savefig(plot_fname, bbox_inches="tight")
