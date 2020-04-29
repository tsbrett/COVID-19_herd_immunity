import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import models.helper as h


params = h.get_params(None)
n_ages = params["n_ages"]

ts_df = pd.read_csv(h.with_fatigue_filepath)
ts_df_nf = pd.read_csv(h.no_fatigue_filepath)

exp_design = pd.read_csv(h.exp_design_filepath, index_col=0)


#%%
def construct_ts_max(df):
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
    ts_total = df.groupby("run")["60+_cases"].sum() / (params["N"][-3:].sum())
    max_df["60+_total"] = ts_total

    max_df = max_df.sort_values("Intervention strategy")

    max_df["hosp_burden"] = df.groupby("run")["hosp_burden"].max()
    return max_df


ts_max = construct_ts_max(ts_df)
ts_max_nf = construct_ts_max(ts_df_nf)


#%%
# Check mitigation but not suppression ranges
hosp_cap = 1e5
int_strat = 1
for int_strat, g in ts_max_nf.groupby("Intervention strategy"):
    tt2 = g[g["R0"] > 1]
    si_range = tt2[tt2["hosp_burden"] < hosp_cap]["si_eff"]
    print(int_strat, si_range.min(), si_range.max())

#%%
use_thin = True

fig_width = 3.4252 if use_thin else 5.2

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(fig_width, 5.2))

fontsize = 8
label_fontsize = 8
fig.subplots_adjust(left=0.185, bottom=0.23,
                    right=0.96, top=0.92, wspace=0.4, hspace=0.6)
legend_loc = (-0.58, -2)

axes[0, 0].text(0.5, 0.84, "Without social\ndistancing fatigue",
                ha='center', fontsize=label_fontsize)
axes[0, 1].text(0.5, 0.84, "With social\ndistancing fatigue",
                ha='center', fontsize=label_fontsize)


for j, ts_m in enumerate([ts_max_nf, ts_max]):
    for _, g in ts_m.groupby("Intervention strategy"):
        tt = g.sort_values("si_eff")
        col = tt["color"].iloc[0]
        name = tt["name"].iloc[0]

        # crude hack to fix legend
        if name == 'SI and 60+ social distancing (SD)':
            name = 'SI and 60+ social\ndistancing (SD)'

        ax = axes[0, j]
        ax.plot(tt["si_eff"], tt["60+_total"], color=col,
                label=name)
        ax.set_yticks(np.linspace(0, 0.6, 4))

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
           "Peak day in 60+", "Peak hospital\nburden (Log$_{10}$)"]

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


axes[-1, 0].legend(frameon=False, ncol=2, fontsize=8, loc=legend_loc)


fig.savefig("./figures/fig_mitigation_summary.pdf")
# plt.show()
