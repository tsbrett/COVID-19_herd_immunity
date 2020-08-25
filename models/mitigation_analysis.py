import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import models.helper as h


#%%
def construct_ts_max(filepath, par):

    exp_design = pd.read_csv(h.exp_design_filepath, index_col=0)

    df = pd.read_csv(filepath)

    df["70+_cases"] = df.groupby("run")["C14"].diff()
    df["C60+"] = df[["C12", "C13", "C14"]].sum(axis=1)
    df["60+_cases"] = df.groupby("run")["C60+"].diff()

    max_indices = df.groupby("run")["60+_cases"].idxmax()

    max_df = df.loc[max_indices, ["time", "60+_cases", "run"]]
    max_df.index = max_df["run"]

    max_df = pd.concat([max_df, exp_design], axis=1)

    # Self-isolation effectiveness
    max_df["si_eff"] = 1-max_df.apply(h.get_self_isolation_factor, axis=1)

    # Total cases 60+
    ts_total = df.groupby("run")["60+_cases"].sum() / (par["N"][-3:].sum())
    max_df["60+_total"] = ts_total

    max_df = max_df.sort_values("Intervention strategy")

    max_df["hosp_burden"] = df.groupby("run")["hosp_burden"].max()

    # Drop strategy not plotted
    max_df = max_df[max_df["name"] != "SI and no 60+ contacts"]
    return max_df


def print_mitigation_ranges(ts_max_nf, hosp_cap=1e5):
    # Check mitigation but not suppression ranges
    for int_strat, g in ts_max_nf.groupby("Intervention strategy"):
        tt2 = g[g["R0"] > 1]
        si_range = tt2[tt2["hosp_burden"] < hosp_cap]["si_eff"]
        print(int_strat, si_range.min(), si_range.max())


def plot_mitigation_summary(ts_max_nf, ts_max):
    use_thin = True

    fig_width = 3.4252 if use_thin else 5.2

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(fig_width, 4.3))

    fontsize = 8
    label_fontsize = 8
    fig.subplots_adjust(left=0.2, bottom=0.21,
                        right=0.96, top=0.905, wspace=0.3, hspace=0.8)
    legend_loc = (-0.59, -2)

    axes[0, 0].text(0.5, 1.15, "Without social\ndistancing fatigue",
                    ha='center', fontsize=label_fontsize)
    axes[0, 1].text(0.5, 1.15, "With social\ndistancing fatigue",
                    ha='center', fontsize=label_fontsize)

    for j, ts_m in enumerate([ts_max_nf, ts_max]):
        for _, g in ts_m.groupby("Intervention strategy"):
            tt = g.sort_values("si_eff")
            col = tt["color"].iloc[0]
            name = tt["name"].iloc[0]

            # crude hack to fix legend
            if name == 'SI and 60+ social distancing (SD)':
                name = 'SI and 60+ social\ndistancing (SD)'

            # crude hack to fix legend
            # if name == 'SI; 0-24, 25-59 and 60+ SD':
            #     name = 'SI; 0-24, 25-59\n and 60+ SD'

            ax = axes[0, j]
            ax.plot(tt["si_eff"], tt["60+_total"], color=col,
                    label=name)
            ax.set_yticks(np.linspace(0, 0.8, 5))

            ax = axes[1, j]
            ax.plot(tt["si_eff"], np.log10(tt["60+_cases"]), color=col,
                    label=name)
            ax.set_yticks(np.linspace(2, 6, 5))

            ax = axes[2, j]
            ax.plot(tt["si_eff"], tt["time"], color=col,
                    label=name)
            ax.set_yticks(np.linspace(0, 500, 6))

            ax = axes[3, j]
            ax.plot(tt["si_eff"], np.log10(tt["hosp_burden"]), color=col,
                    label=name)
            ax.set_yticks(np.linspace(3, 6, 4))

    ylabels = ["Fraction of\n60+ exposed", "Peak daily 60+\ncases (Log$_{10}$)",
               "Peak day\nin 60+", "Peak hospital\nburden (Log$_{10}$)"]

    for i, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(ylabels[i], fontsize=label_fontsize)

    for i, ax in enumerate(axes[:, 1]):
        ax.set_yticklabels([])

    panel_titles = ["a", "e", "b", "f", "c", "g", "d", "h"]

    for i, ax in enumerate(axes.flatten()):
        ax.set_title(panel_titles[i] + ")", loc="left", fontsize=label_fontsize)
        sns.despine(ax=ax, trim=True, offset=5)
        ax.tick_params(labelsize=fontsize)

        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 5))

    for ax in axes[:-1, :].flatten():
        ax.set_xticklabels([])

    for ax in axes[-1, :]:
        ax.set_xlabel("Self-isolation\neffectiveness", fontsize=label_fontsize)

    axes[-1, 0].legend(frameon=False, ncol=3, fontsize=6, loc=legend_loc,
                       handlelength=1, columnspacing=0.8)

    fig.savefig("./figures/fig_mitigation_summary.pdf")


# Test
# ts_max = construct_ts_max(h.with_fatigue_filepath, par=params)
# ts_max_nf = construct_ts_max(h.no_fatigue_filepath, par=params)
# print_mitigation_ranges(ts_max_nf, hosp_cap=1e5)
# plot_mitigation_summary(ts_max_nf, ts_max)
# plt.show()