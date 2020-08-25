import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import models.helper as h
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

def plot_summary_figure(par):
    n_ages = par["n_ages"]
    stay_duration = par["stay_duration"]
    hosp_cap = 17800
    Ntot = par["N"].sum()


    ts_df = pd.read_csv(h.no_fatigue_filepath)
    exp_df = pd.read_csv(h.exp_design_filepath, index_col=0)


    exp_df["max_burden"] = ts_df.groupby("run")["hosp_burden"].max()

    ts_df["S_tot"] = ts_df[["S" + str(i) for i in range(n_ages)]].sum(axis=1)

    R0_relaxed = exp_df.loc[0, "R0"]
    hi_threshold = (1/R0_relaxed)*par["N"].sum()

    exp_df["name"] = exp_df["name"].replace(to_replace="SI and 60+ social "
                                                       "distancing (SD)",
                                            value="SI and 60+ social\n"
                                                  "distancing (SD)")

    exp_df["name"] = exp_df["name"].replace(to_replace="SI; 0-24, 25-59 "
                                                       "and 60+ SD",
                                            value="SI; 0-24, 25-59\nand 60+ SD")

    eg = exp_df.groupby("name")
    eg_max = eg.apply(lambda x: x[x["max_burden"] < hosp_cap]["R0"].max())

    exp_df_f = exp_df[exp_df["Intervention strategy"] != 5]
    exp_df_f = exp_df_f.reset_index()
    select_runs = exp_df[((exp_df["comp_factor"] * 100 % 10) == 0)].index
    ts_df_light = ts_df[ts_df["run"].isin(select_runs)]


    # Time to hospitalise
    C_cols = ["C" + str(i) for i in range(15)]
    ts_df["Ctot"] = ts_df[C_cols].sum(axis=1)
    ts_df["hosp_rate"] = ts_df["new_hosp"]/ts_df["Ctot"]

    # # maximum hospital rate
    max_hosp_rate = ts_df.groupby("run")["hosp_rate"].max().sort_values()

    print("Max hosp rate for each strategy: ")
    print(max_hosp_rate[np.arange(0,6)])
    # print(max_hosp_rate)
    #
    # max_hosp_rate.hist()
    # plt.show()

    # Time to reach herd immunity at max susceptible depletion rate
    min_time_to_hi = (par["N"].sum() - hi_threshold) \
                     * stay_duration * max_hosp_rate[1]/hosp_cap
    print("min_time_to_hi", min_time_to_hi)

    print("max R0:", eg_max.max())


    def col_function(x):
        hosp_max = x["hosp_burden"].max()
        if (x["S_tot"].iloc[-1] < hi_threshold) & (hosp_max < hosp_cap):
            c = "green"
        elif hosp_max < hosp_cap:
            c = h.colours[9]
        else:
            c = h.colours[12]
        return c


    use_thin = True

    fig_width = 3.4252 if use_thin else 5.2
    #fig_width= 8

    #fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_width, 5.0))

    fig = plt.figure(figsize=(fig_width, 3.5))
    gs = GridSpec(6, 1)
    axes = np.array([plt.subplot(gs[0:3, 0]), plt.subplot(gs[3:, 0])])


    fontsize = 8
    label_fontsize = 8
    legend_loc0 = (0.52, -0.0)
    legend_loc1 = (0.1, -0.05)
    fig.subplots_adjust(left=0.18, bottom=0.13,
                        right=0.96, top=0.95, wspace=0.4, hspace=40)
    # fig.subplots_adjust(left=0.17, bottom=0.09,
    #                     right=0.96, top=0.96, wspace=0.5, hspace=0.5)

    ax = axes[0]
    for _, g in exp_df.groupby("Intervention strategy"):
        col = g["color"].iloc[0]
        name = g["name"].iloc[0]
        if name == "SI and no 60+ contacts":
            name = "SI and no\n60+ contacts"
            ax.plot(g["R0"], np.log10(g["max_burden"]), color=col, label=name)
        if name == "Self-isolation (SI)":
            ax.plot(g["R0"], np.log10(g["max_burden"]), color=col, label=name)


    ax.axvline(eg_max.max(), color="grey", linestyle="--")

    ax.fill_between([1, eg_max.max()], [3, 3], [np.log10(hosp_cap),
                                                np.log10(hosp_cap)],
                    facecolor="#65C8D0", alpha=0.4)

    ax.axvline(1, linestyle="--", color="grey")
    ax.axhline(np.log10(hosp_cap), linestyle="--", color="grey", xmax=2.3/2.5)
    ax.text(2.05, 4.3, s="Hospital\ncapacity\nexceeded", color="grey", ha="center",
            va="bottom", fontsize=label_fontsize)
    ax.text(0.5, 4.2, s="Suppression\nachieved", color="grey", ha="center",
            va="top", fontsize=label_fontsize)
    ax.set_xlabel("Reproductive number", fontsize=label_fontsize)
    ax.set_ylabel("Peak hospital\nburden (Log$_{10}$)", fontsize=label_fontsize)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(3.0, 6.)
    ax.set_yticks(np.linspace(3, 6, 4))
    ax.legend(frameon=False, fontsize=6, loc=2,
              handlelength=1, columnspacing=0.5, facecolor="white")

    df_list = [ts_df_light] #todo: fix handeling of this
    for i, df in enumerate(df_list):
        ax = axes[1]
        td_g = df.groupby("run")
        for run, g in td_g:
            col = col_function(g)
            ax.plot(g["hosp_burden"]/1e5, g["S_tot"]/1e6, color=col, alpha=0.8)

        ax.axhline(hi_threshold/1e6, linestyle="--", color="grey")
        ax.axvline(hosp_cap/1e5, linestyle="--", color="grey")
        ax.set_xlabel("Hospital burden (hundred thousands)",
                      fontsize=label_fontsize)
        ax.set_ylabel("Susceptible individuals\n(millions)",
                      fontsize=label_fontsize)

        ax.set_xlim(0, 8)
        ax.set_ylim(0, 67)
        ax.set_yticks(np.linspace(0, 60, 4))

        ax.text(0.6, 62, s="Hospital capacity", color="grey",
                fontsize=label_fontsize)
        ax.text(8, 32, s="Herd immunity\nthreshold", color="grey",
                ha="right", fontsize=label_fontsize)

    # ax.plot(df_seir["hosp_burden"]/1e5, df_seir["S"]/1e6, color=h.colours[2])

    legend_elements = [Line2D([0], [0], color=h.colours[9],
                              label='Below capacity'),
                       Line2D([0], [0], color=h.colours[12],
                              label='Exceeds capacity'),
                       # Line2D([0], [0], color=h.colours[2],
                       #        label='At capacity'),
                       ]

    ax.legend(handles=legend_elements, ncol=2, loc=legend_loc1, fontsize=6,
              handlelength=1, columnspacing=2, frameon=False)




    titles = ["a)", "b)", "c)", "d)", "e)"]
    for i, ax in enumerate(axes):
        sns.despine(ax=ax, offset=5, trim=True)
        ax.set_title(titles[i], loc="left", fontsize=label_fontsize)
        ax.tick_params(labelsize=fontsize)


    fig.savefig("./figures/fig_summary.pdf")

# plot_summary_figure(par=params)
# plt.show()
