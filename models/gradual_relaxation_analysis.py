import models.helper as h
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_gradual_relaxation(par):

    stay_duration = par["stay_duration"]
    n_ages = par["n_ages"]

    # Read in exp_design matrix. Simulation script uses row 54
    exp_design = pd.read_csv(h.exp_design_filepath, index_col=0)

    # To get HI threshold if all SD but SI remains at 0.1
    R0_relaxed = exp_design.loc[60, "R0"]  # todo: not robust to no. of strats

    # HI threshold for *susceptible* population
    hi_threshold = (1/R0_relaxed)*par["N"].sum()

    # Read in simulation results
    out_df = pd.read_csv(h.gradual_relaxation_filepath)

    # Total susceptible and new cases
    C_cols = ["C"+str(y) for y in range(0, n_ages)]
    S_cols = ["S"+str(y) for y in range(0, n_ages)]
    out_df["Stot"] = out_df[S_cols].sum(axis=1)
    out_df["Ctot"] = out_df[C_cols].sum(axis=1)

    # Daily fraction of new cases hospitalised
    out_df["hosp_rate"] = out_df["new_hosp"]/out_df["Ctot"]

    # Total cumulative fatalities
    out_df["fatalities"] = out_df[['<60_fatal', '>60_fatal']].sum(axis=1)

    # maximum hospital burden
    print("Max. hosp. burden:", out_df.groupby("run")["hosp_burden"].max())

    # maximum hospital rate
    max_hosp_rate = out_df.groupby("run")["hosp_rate"].max()
    print("Max. hosp. rate:", max_hosp_rate)

    # Time to reach herd immunity at max susceptible depletion rate using social
    # distancing

    hosp_capacity = 1e5
    min_time_to_hi = (par["N"].sum() - hi_threshold) * stay_duration\
        * max_hosp_rate[1] / hosp_capacity
    print("Hospital capacity:", hosp_capacity)
    print("Minimum time to herd immunity:", min_time_to_hi)
    # Final fatalities
    print("Final fatalities:", out_df.groupby("run")["fatalities"].max())

    colours = h.colours

    runs = out_df["run"].unique()
    runs_names = dict(zip(runs, ["No intervention",
                                 "SI and 60+ social\ndistancing (SD)",
                                 "0-59 SD relaxed over 100 days",
                                 "0-59 SD relaxed over 200 days",
                                 "0-59 SD relaxed over 300 days"]))

    # HI threshold for *susceptible* population
    hi_threshold = (1/R0_relaxed)*par["N"].sum()

    runs_cols = dict(
        zip(runs, [colours[11], colours[8], colours[5], colours[12],
                   colours[0]]))

    use_thin = False

    fig_width = 3.4252 if use_thin else 5.2

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(fig_width, 5.0))

    fig.subplots_adjust(left=0.18, bottom=0.09,
                        right=0.92, top=0.96, wspace=0.4, hspace=0.4)
    legend_loc = (0.3, 0.25)
    fontsize = 8
    label_fontsize = 8

    for run, g in out_df.groupby("run"):

        ax = axes[0]
        ax.plot(g["time"], g["hosp_burden"]/1e5, label=runs_names[run],
                color=runs_cols[run])

        ax = axes[1]
        ax.plot(g["time"], g["fatalities"]/1e5, label=run,
                color=runs_cols[run])

        ax = axes[2]
        ax.plot(g["time"], g["Stot"]/1e6, label=run,
                color=runs_cols[run])

    axes[0].set_title("a)", loc="left", fontsize=label_fontsize)
    axes[0].set_ylabel("Hospital burden\n(hundred thousands)",
                       fontsize=label_fontsize)
    axes[0].set_yticks(np.linspace(0, 12, 5))
    axes[0].legend(frameon=False,  fontsize=8, loc=1) #legend_loc)
    axes[0].axvline(par["tint"], color="grey", linestyle="--")

    axes[1].set_title("b)", loc="left", fontsize=label_fontsize)
    axes[1].set_ylabel("Fatalities\n(hundred thousands)",
                       fontsize=label_fontsize)

    axes[1].set_yticks([0, 1, 2, 3, 4])

    axes[2].set_title("c)", loc="left", fontsize=label_fontsize)
    axes[2].set_xlabel("Day", fontsize=label_fontsize)
    axes[2].set_ylabel("Susceptible population\n(millions)",
                       fontsize=label_fontsize)
    axes[2].set_yticks([10, 20, 30, 40, 50, 60, 70])
    axes[2].text(-0, 48, "Herd\nimmunity\nthreshold", fontsize=8, va="top")
    axes[2].axhline(hi_threshold/1e6, color="grey", linestyle="--")

    for ax in axes:
        ax.set_xlim(0, 300)
        sns.despine(ax=ax, trim=True, offset=5)
        ax.tick_params(labelsize=fontsize)

    fig.savefig("./figures/fig_gradual_relaxation.pdf")
    #plt.show()


# plot_gradual_relaxation(params)
