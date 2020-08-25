import models.helper as h
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_introduction_figure(par):

    out_df = pd.read_csv(h.with_fatigue_filepath)

    exp_design = pd.read_csv(h.exp_design_filepath, index_col=0)

    n_ages = par["n_ages"]
    age_groups = par["age_groups"]

    pop = pd.Series(par["N"], index=age_groups)

    cols = dict(zip(age_groups, h.colours))

    C_cols = ["C" + str(y) for y in range(0, n_ages)]

    Ig = out_df.groupby("run")
    Ig.index = Ig["time"]

    new_cols = {"0-19": age_groups[0:4],
                "20-39": age_groups[4:8],
                "40-59": age_groups[8:12],
                "<60": age_groups[:12],
                "≥60": age_groups[12:]}

    for i, c in new_cols.items():
        pop[i] = pop[c].sum()

    cols["0-19"] = h.colours[0]
    cols["20-39"] = h.colours[8]
    cols["40-59"] = h.colours[11]
    cols["≥60"] = h.colours[12]
    cols["<60"] = h.colours[5]

    # pick self-isolation effectiveness = 0.5, equivalent to 0.75 "engagement"
    # with 0.66 symptomatic

    exp_comp = exp_design["comp_factor"]
    exp_is = exp_design["Intervention strategy"]
    i0 = exp_design.loc[(exp_comp == 0.) & (exp_is == 0), :].index[0]
    i1 = exp_design.loc[(exp_comp == 0.2) & (exp_is == 1), :].index[0]
    i2 = exp_design.loc[(exp_comp == 0.2) & (exp_is == 3), :].index[0]

    runs = [i0, i1, i2]

    for run in runs:
        g = Ig.get_group(run)
        prop_inf = (g[C_cols].sum(axis=1) / pop[["<60", "≥60"]].sum()).max()
        no_fatal = g[['<60_fatal', '>60_fatal']].max()
        print("Run = %d, Proportion infected = %f, No. fatalities = %s"
              % (run, prop_inf, str(no_fatal.values)))

    fig, axes = plt.subplots(3, 3, figsize=(7, 3.))

    fontsize = 8
    label_fontsize = 8

    for i, run in enumerate(runs):
        g = Ig.get_group(run)
        I_out = g[C_cols]
        f_out = g[['<60_fatal', '>60_fatal']]
        t_out = g["time"]
        I_df = I_out.copy()
        I_df.index = t_out
        I_df.columns = age_groups

        f_df = f_out.copy()
        f_df.columns = ["<60", "≥60"]
        f_df.index = t_out

        I_df2 = pd.DataFrame()

        for j, c in new_cols.items():
            I_df2[j] = I_df[c].sum(axis=1)

        for c, ts in I_df2[["0-19", "20-39", "40-59", "≥60"]].iteritems():

            axes[0, i].plot(ts.diff() / 1e3, label=c, color=cols[c])
            axes[1, i].plot(ts/pop[c], label=c, color=cols[c])

        for c, ts in f_df.iteritems():
            axes[2, i].plot(ts/1e3, label=c, color=cols[c])

    for ax in axes[0, :]:
        ax.set_ylim(-5, 700)
        ax.set_yticks([0, 200, 400, 600])
    for ax in axes[1, :]:
        ax.set_ylim(-0.01, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for ax in axes[2, :]:
        ax.set_ylim(-3, 415)
        ax.set_yticks([0, 100, 200, 300, 400])


    axes[0, 0].set_ylabel("Daily new cases \n (thousands)",
                          fontsize=label_fontsize)
    axes[2, 0].set_ylabel("Cumulative\nfatalities\n(thousands)",
                          fontsize=label_fontsize)
    axes[1, 0].set_ylabel("Cumulative\nproportion\nexposed",
                          fontsize=label_fontsize)


    for ax in axes[-1, :]:
        ax.set_xlabel("Day", fontsize=label_fontsize)

    alphabet = ["a", "d", "g", "b", "e", "h", "c", "f", "i"]

    for i, ax in enumerate(axes.flatten()):
        ax.set_xlim(0, 250)
        ax.set_xticks(np.linspace(0, 250, 6))
        ax.tick_params(labelsize=fontsize)
        sns.despine(ax=ax, trim=True, offset=5)
        ax.set_title(alphabet[i] + ")", loc="left", fontsize=label_fontsize)


    for ax in axes[:, 1:].flatten():
        ax.set_yticklabels(labels=[])
    for ax in axes[:-1, :].flatten():
        ax.set_xticklabels(labels=[])

    for ax in axes[1:, 1:].flatten():
        ax.axvline(par["tint"], linestyle="--", color="grey")

    for ax in axes[1:, 2]:
        ax.axvline(par["tint2"], linestyle="--", color="grey")

    for ax in axes[0, 1:]:
        ax.axvline(par["tint"], linestyle="--", color="grey", ymax=600/700)

    axes[0, 2].axvline(par["tint2"], linestyle="--", color="grey", ymax=600/700)

    axes[0,0].legend(ncol=2, fontsize=fontsize, facecolor='white',
                     handlelength=1, edgecolor='white', framealpha=1.0,
                     columnspacing=0.5, loc=(0.5, 0.1))
    axes[1,0].legend(ncol=2, fontsize=fontsize, facecolor='white',
                     handlelength=1, frameon=False,
                     columnspacing=0.5, loc=(0.5, -0.05))
    axes[2,0].legend(ncol=1, fontsize=fontsize, facecolor='white',
                     handlelength=1, edgecolor='white', framealpha=1.0)

    axes[0, 1].text(x=par["tint"]+4, y=600, s="Control measures\nstart",
                    va="top",
                    color="grey", fontsize=fontsize)
    axes[0, 2].text(x=par["tint"]+4, y=600, s="Control\nmeasures\nstart",
                    va="top",
                    color="grey", fontsize=fontsize)
    axes[0, 2].text(x=par["tint2"]+4, y=600, s="Schools\nre-open",
                    va="top",
                    color="grey",
                    fontsize=fontsize)

    plt.subplots_adjust(left=0.12, bottom=0.15,
                        right=0.98, top=0.94, wspace=0.3, hspace=0.65)

    fig.savefig("./figures/fig_introduction.pdf")
    # plt.show()

# plot_introduction_figure(par=params)