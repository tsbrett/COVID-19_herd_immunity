import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import models.helper as h
import seaborn as sns
from matplotlib.patches import Patch


# Colours for plotting
cols = ["#8C9EDE", "#65C8D0", "#00767B", "#285C9F", "#04375E"]


def get_exp_results(par):



    ts_df = pd.read_csv(h.no_fatigue_filepath)
    i_tot_at_int = ts_df[ts_df["time"] == par["tint"]].iloc[0]["I_tot"]
    i_threshold = i_tot_at_int / 100
    df_f = ts_df[ts_df["I_tot"] > i_threshold]
    df_f = df_f[df_f["time"] > par["tint"]]
    t_end = df_f.groupby("run")["time"].max()

    exp_results = pd.read_csv(h.exp_design_filepath, index_col=0)

    exp_results["t_end"] = t_end
    exp_results["t_dur"] = t_end - par["tint"]

    # Get effect of social distancing on R0
    sup_df = h.get_social_distance_strats()
    sup_df["prob_s"] = 0
    sup_df["comp_factor"] = 0

    def int_wrap(x):
        return h.get_intervention_R0(par=par, var_par=x, sd_on=True)

    sup_df["R0"] = sup_df.apply(int_wrap, axis=1)

    # Print time to achieve 100-fold reduction
    t_60 = exp_results[exp_results["t_end"] <
                       (60 + par["tint"])].groupby("name")["si_eff"].min()

    print(t_60)

    # Drop strategy not plotted
    exp_results = exp_results[exp_results["name"] != "SI and no 60+ contacts"]

    sup_df = sup_df[sup_df["name"] != "SI and no 60+ contacts"]

    exp_results["col"] = exp_results["Intervention strategy"]\
        .apply(lambda x: cols[x])

    return sup_df, exp_results


def plot_suppression_results(par):

    sup_df, exp_results = get_exp_results(par=par)

    use_thin = True
    fig_width = 3.4252 if use_thin else 5.2

    fig, axes = plt.subplots(nrows=3, figsize=(fig_width, 4.5))

    fontsize = 8
    label_fontsize = 8
    fig.subplots_adjust(left=0.16, bottom=0.1,
                        right=0.97, top=0.95, wspace=0.5, hspace=0.6)

    ax = axes[0]
    comp_arr = np.arange(0.01, 1.01, 0.01)
    r_prev = 1.1
    text_locs = [(0.64, 0.9), (0.25, 0.73), (0.35, 0.58), (0.43, 0.43),
                 (0.66, 0.22)]

    for i, r in sup_df.iterrows():
        name = r["name"]
        if name == 'SI and 60+ social distancing (SD)':
            name = 'SI and 60+ social\ndistancing (SD)'
        if name == 'SI; 0-24, 25-59 and 60+ SD':
            name = 'SI; 0-24, 25-59\nand 60+ SD'

        r_arr = (1 - 1 / r["R0"]) / comp_arr
        ax.plot(comp_arr, r_arr, color=cols[i])
        ax.fill_between(comp_arr, r_arr, r_prev, alpha=0.9,
                        color=cols[i])
        r_prev = r_arr
        ax.text(*text_locs[i], s=name, color="white", fontsize=6)

    legend_elements = [Patch(facecolor='lightgrey', edgecolor='lightgrey',
                             label="Suppression \npossible ($R_0 < 1$)")]

    ax.legend(handles=legend_elements, loc='lower left', frameon=False,
              fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Self-isolation observance rate", fontsize=label_fontsize)
    ax.set_ylabel("Proportion symptomatic", fontsize=label_fontsize)
    ax.set_title("a)", loc="left", fontsize=label_fontsize)

    ax = axes[1]

    for _, g in exp_results.groupby("Intervention strategy"):
        tt = g.sort_values("si_eff")
        col = tt["col"].iloc[0]
        name = tt["name"].iloc[0]
        ax.plot(tt["si_eff"], tt["R0"], color=col, label=name)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.5)
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_title("b)", loc="left", fontsize=label_fontsize)
    ax.set_xlabel("Self-isolation effectiveness", fontsize=label_fontsize)
    ax.set_ylabel("Reproductive number", fontsize=label_fontsize)
    ax.text(1, 1.05, "Suppression\nthreshold", color="grey", ha="right",
            fontsize=fontsize)
    ax.text(1, 2.35, "Baseline $R_0$", color="grey", fontsize=fontsize,
            ha="right")
    ax.axhline(1, linestyle="--", color="grey")
    ax.axhline(exp_results["R0"].max(), linestyle="--", color="grey")

    ax = axes[2]
    exp_results_filter = exp_results.loc[exp_results["R0"] < 0.9, :]
    exp_results_filter = exp_results_filter.loc[exp_results["t_dur"] < 465, :]

    for _, g in exp_results_filter.groupby("Intervention strategy"):

        tt = g.sort_values("si_eff")
        col = tt["col"].iloc[0]
        name = tt["name"].iloc[0]
        if name == 'SI and 60+ social distancing (SD)':
            name = 'SI and 60+ social\ndistancing (SD)'
        if name == 'SI; 0-24, 25-59 and 60+ SD':
            name = 'SI; 0-24, 25-59 and 60+ SD'

        ax.plot(tt["si_eff"], tt["t_dur"], color=col, label=name)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 150)
    ax.set_yticks(np.linspace(0, 150, 6))
    ax.set_title("c)", loc="left", fontsize=label_fontsize)
    ax.set_xlabel("Self-isolation effectiveness", fontsize=label_fontsize)
    ax.set_ylabel("Days to suppression", fontsize=label_fontsize)
    # note in this context suppression =  a hundred-fold reduction

    #axes[2].legend(frameon=False, ncol=2, fontsize=8, loc=(-0.2, -0.96))
    axes[2].legend(ncol=1, fontsize=6, frameon=False,
                   handlelength=1, columnspacing=0.5, loc=(-0.01,-0.05))
    for ax in axes:
        sns.despine(ax=ax, offset=5, trim=True)
        ax.tick_params(labelsize=fontsize)

    fig.savefig("./figures/fig_suppression_results.pdf")

# plot_suppression_results(par=params)
# plt.show()