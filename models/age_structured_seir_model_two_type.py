import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import models.helper as h
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

n_cores = 7


def get_q_factor(S, E, par):
    beta = par["beta"]
    N = par["N"]
    g = par["g"]
    rho = par["rho"]
    cm = par["cm"]
    hc = par["hosp_cap"]
    stay_duration = par["stay_duration"]
    hosp_rate = par["hosp_rate"]
    hosp_cases = E[0]*rho*stay_duration*hosp_rate
    # Implement using "events"
    q = np.max([0, 1 - N[0]/(cm[0, 0]*S[0]*beta[0]/g)]) if hosp_cases > hc else 0

    return q

def get_min_time_to_herd_immunity_approx(par):

    # Time to reach herd immunity at max susceptible depletion rate
    beta = par["beta"]
    N = par["N"]
    g = par["g"]
    cm = par["cm"]
    hi_threshold = 1 - N[0] / (cm[0, 0] * N[0] * beta[0] / g)

    min_time_to_hi = N[0]*hi_threshold \
                     * par["stay_duration"] * par["hosp_rate"]/par["hosp_cap"]
    return min_time_to_hi



def hosp_control_cm(S, E, par):

    E_sum = E.sum(axis=0)  # sum over stages
    cm = par["cm"]
    hc = par["hosp_cap"]
    stay_duration = par["stay_duration"]
    hosp_rate = par["hosp_rate"]
    q = get_q_factor(S, E_sum, par)
    cm2 = np.zeros(cm.shape)
    cm2[0, 0] = (1 - q) * cm[0, 0]
    return cm2


def solve_SIRS_model(par):


    initial_cond = h.set_initial_conditions(par)
    # End all interventions not affecting older age classes

    def model_wrapper(t, y, par):

        S, E, I, R, C = h.split_seir_compartments(y, par)
        par["cm_sd"] = hosp_control_cm(S, E, par)
        S, E, I, R, C = h.age_structured_SEIR_model(S=S, E=E, I=I, R=R,
                                                    C=C, par=par)
        y_new = np.concatenate((S, E.flatten(), I.flatten(), R, C))
        return y_new

    sol = solve_ivp(fun=model_wrapper, t_span=[0, par["tmax"]],
                    y0=initial_cond, method="RK45",
                    t_eval=np.arange(0, par["tmax"]+1, 1),
                    args=(par,))

    ret_df = h.convert_to_data_frame(output=sol, par=par)
    return ret_df


def tng_wrapper(x, par, only_first=True):
    S = x[["S" + str(i) for i in range(par["n_ages"])]]
    if only_first:
        cm = np.zeros(par["cm"].shape)
        cm[0, 0] = par["cm"][0, 0]
    else:
        cm = par["cm"]
    return h.get_TNG(cm=cm, beta=par["beta"], g=par["g"], N=par["N"], S=S)[0]


def two_type_df_wrapper(par, var, v):

    par_sim = {**par, **{var: v}}
    df = solve_SIRS_model(par=par_sim) # update var to have value v

    df["Ctot"] = df[["C0", "C1"]].sum(axis=1)
    df["hosp_burden"] = df["Ctot"].diff() * par_sim["hosp_rate"] \
                        * par_sim["stay_duration"]
    df["Reff_single"] = df.apply(tng_wrapper, par=par_sim, axis=1)
    df["Reff_full"] = df.apply(tng_wrapper, par=par_sim,
                               only_first=False, axis=1)
    df["control"] = df.apply(lambda x: get_q_factor(S=x[["S0", "S1"]],
                                                    E=x[["E0", "E1"]],
                                                    par=par_sim), axis=1)
    df[var] = v
    return df


def run_seir_varying_control_simulations(var, R0):

    # convert to two type:
    index_map = [np.arange(0, 12), np.arange(12, 15)]
    # todo:  may be computationally smart to move this outside
    par = h.get_two_type_params(index_map=index_map, R0=R0)
    #par["stay_duration"] = 6
    par["tmax"] = 365*10 # todo: check this is long enough
    par["hosp_cap"] = 17800
    # par["hosp_rate"] = 0.0368

    hc_range = np.arange(2200, 60000, 1200)
    hc_range[28] = 35600 # add exactly double 17800 to list
    sd_range = np.arange(1, 25, 1)

    if var == "hosp_cap":
        var_range = hc_range

    if var == "stay_duration":
        var_range = sd_range



    start_time = time.time()
    # tt = pd.concat([get_two_type_df(hosp_cap=h) for h in
    #                 np.arange(0.1, 1.1, 0.1)*1e5])
    sim_df = Parallel(n_jobs=n_cores)(delayed(two_type_df_wrapper)(par, var, v)
                                      for v in var_range)
    sim_df = pd.concat(sim_df)
    sim_df = sim_df.sort_values([var, "time"])
    print("--- %s seconds ---" % (time.time() - start_time))

    def get_ctl_dur(df):
        df_f = df[df["control"] > 0]
        return df_f.index.max() - df_f.index.min()

    def get_final_full_R0(df):
        return df["Reff_full"].iloc[-1]

    def get_final_sgl_R0(df):
        return df["Reff_single"].iloc[-1]

    def get_S0_start(df):
        df_f = df[df["control"] > 0]
        return df_f.loc[df_f.index.min(), "S0"]

    out_g = sim_df.groupby(var)
    out_stats = pd.DataFrame({"control_duration": out_g.apply(get_ctl_dur),
                              "final_full_R0": out_g.apply(get_final_full_R0),
                              "final_single_R0": out_g.apply(get_final_sgl_R0)})

    out_stats["full_hi"] = out_stats["final_full_R0"] < 1
    out_stats["S0"] = out_g.apply(get_S0_start)

    def mt_wrapper(x):
        par_x = {**par, **{var: x.name, "S0": x["S0"]}}
        return get_min_time_to_herd_immunity_approx(par_x)

    out_stats["approx_time_to_hi"] = out_stats.apply(mt_wrapper, axis=1)

    return sim_df, out_stats


def plot_optimal_strategy(out_df, out_stats):

    pal = [h.colours[11], h.colours[0]]
    pal2 = [h.colours[8], h.colours[12]]
    fontsize = 8
    label_fontsize = 8
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 2.5))

    fig.subplots_adjust(left=0.09, bottom=0.18,
                        right=0.97, top=0.925, wspace=0.3, hspace=0.67)

    out_g = out_df[out_df["hosp_cap"].isin([17800, 35600])].groupby("hosp_cap")

    ax = axes[0, 0]
    for i, (hc, g) in enumerate(out_g):
        g = g.sort_index()
        ax.plot(g.index, g["hosp_burden"]/1000, alpha=0.7, color=pal[i],
                label=hc/1000)

    ax.set_ylabel("Hospitalized cases\n(thousands)", fontsize=label_fontsize)
    ax.set_xlim(0, 600)
    ax.set_ylim(-2, 50)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.legend(frameon=False, fontsize=fontsize,
              title="Hospital capacity\n(thousand beds)",
              title_fontsize=fontsize,
              loc=(0.66, 0.4),
              facecolor="white")

    ax = axes[1, 0]
    for i, (hc, g) in enumerate(out_g):
        g = g.sort_index()
        ax.plot(g["control"], alpha=0.7, color=pal[i], label=hc/1000)
    ax.set_ylabel("$<60$ contact\nreduction", fontsize=label_fontsize)
    ax.set_xlabel("Day", fontsize=label_fontsize)
    ax.set_ylim(-0.05, 0.6)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xlim(0, 600)

    ax = axes[0, 1]
    ax.plot(out_stats.index/1e3, out_stats["final_single_R0"], color=pal2[0],
            label="60+ remain isolated")
    ax.plot(out_stats.index/1e3, out_stats["final_full_R0"], color=pal2[1],
            label="60+ cease isolation")
    ax.set_ylabel("Final reproductive\nnumber", fontsize=label_fontsize)
    ax.axhline(1, linestyle="--", color="grey")
    ax.legend(frameon=False, fontsize=fontsize,
              title_fontsize=fontsize,
              loc=(0.5, 0.75),
              facecolor="white")
    ax.set_xlim(0, 60)
    ax.set_ylim(0.65, 1.1)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0, 1.1])

    ax = axes[1, 1]
    out_stats_hi = out_stats[out_stats["full_hi"]]
    out_stats_no_hi = out_stats[~out_stats["full_hi"]]


    ax.plot(out_stats.index/1e3, 12*out_stats["control_duration"]/365,
            color=pal2[0], label="Herd immunity achieved")
    ax.plot(out_stats_no_hi.index/1e3, 12*out_stats_no_hi["control_duration"]/365,
            color=pal2[1], label="Herd immunity not achieved")

    ax.axvline(17.8, linestyle="--", color="grey")  # from data file
    ax.set_ylabel("Control duration\n(months)", fontsize=label_fontsize)
    ax.set_xlabel("Hospital capacity (thousand beds)", fontsize=label_fontsize)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 4*12)
    ax.set_yticks(12*np.array([0, 1, 2, 3, 4]))
    ax.legend(frameon=False, fontsize=fontsize,
              title_fontsize=fontsize,
              loc=(0.4, 0.2),
              facecolor="white")
    ax.text(x=18.8, y=4*12, s="Average UK hospital burden\n(April 2020)",
            color="grey", va="top", ha="left", fontsize=label_fontsize)
    abc = ["a", "c", "b", "d"]
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(abc[i] + ")", loc="left", fontsize=label_fontsize)
        sns.despine(ax=ax, trim=True, offset=5)
        ax.tick_params(labelsize=fontsize)

    fig.savefig("./figures/fig_optimal_strategy.pdf")
    # plt.show()


def get_heatmap_data(R0):
    index_map = [np.arange(0, 12), np.arange(12, 15)]
    par = h.get_two_type_params(index_map=index_map, R0=R0)
    par["hosp_cap"] = 17800  

    dy = 0.001
    dx = 0.5

    sd_range = np.arange(dx, 26, dx)
    hr_range = np.arange(dy, 0.1, dy)

    from itertools import product
    df = pd.DataFrame(list(product(sd_range, hr_range)), columns=['stay_duration',
                                                                  'hosp_rate'])


    def get_time_to_hi_wrapper(x):
        par_x = {**par, **x}
        return get_min_time_to_herd_immunity_approx(par=par_x)*12/365

    df["approx_time_to_hi"] = df.apply(get_time_to_hi_wrapper, axis=1)
    df["hosp_rate"] = df["hosp_rate"].round(3)
    df = df.set_index(["stay_duration", "hosp_rate"])
    df_unstack = df["approx_time_to_hi"].unstack("stay_duration")

    return par, df_unstack


def plot_control_duration_heatmap(par, df):
    dy = np.diff(df.index.values).max()
    dx = np.diff(df.columns.values).max()
    ymin, ymax = df.index.min(), df.index.max()
    xmin, xmax = df.columns.min(), df.columns.max()

    col = h.colours
    pal = [col[10], col[11], col[12], col[13]]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(df, aspect="auto",
                   origin='lower',
                   extent=[xmin-dx/2, xmax+dx/2, ymin-dy/2, ymax+dy/2],
                   cmap=sns.cubehelix_palette(8, as_cmap=True))
    cb = fig.colorbar(im, ax=ax, label="Control duration (months)")
    cb.outline.set_edgecolor('white')


    # ax.plot(xpoints, ypoints)
    ax.set_ylim(ymin-dy/2, ymax+dy/2)
    ax.set_xlim(xmin-dx/2, xmax+dx/2)
    contours = ax.contour(df.columns.values.flatten(),
                           df.index.values.flatten(),
                           df.values, levels=[6, 12, 24], colors='black')

    # ONS seroprevalence estimates
    cnt_level = (1-1/par["R0_baseline"])*np.array([1/0.0864, 1/0.0678, 1/0.0521])
    contours2 = ax.contourf(df.columns.values.flatten(),
                           df.index.values.flatten(),
                           df.values, levels=cnt_level,
                           colors='red', alpha=0.3)

    ax.text(13, 0.0065, "Estimated duration from serology",
            fontsize=8, rotation=-7, color="red", alpha=0.9)


    print("Control duration from serology estimate: ",cnt_level)

    ax.clabel(contours, inline=True, fontsize=8, fmt={6: "6 months",
                                                      12: "12 months",
                                                      24: "24 months"})

    # https://www.medrxiv.org/content/10.1101/2020.04.30.20084780v3
    ax.scatter(5, par["hosp_rate"], color=pal[0], label="Rees et al., 2020")

    # https://www.medrxiv.org/content/10.1101/2020.04.23.20076042v1.full.pdf
    ax.scatter(7, par["hosp_rate"], color=pal[1], label="Docherty et al., 2020")

    #https://www.nejm.org/doi/full/10.1056/NEJMoa2002032
    ax.scatter(12.8, par["hosp_rate"], color=pal[2], label="Guan et al., 2020")

    # https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext
    ax.scatter(24.7, par["hosp_rate"], color=pal[3], label="Verity et al., 2020")
    ax.set_xlabel("Hospital stay duration (days)")
    ax.set_ylabel("Hospitalisation probability")
    ax.set_xticks(np.arange(0,26,5))
    ax.set_yticks(np.arange(0,0.11,0.02))
    sns.despine(ax=ax, offset=5, trim=True)
    ax.legend(frameon=False, loc=5, fontsize=8)
    fig.tight_layout()
    fig.savefig("./figures/fig_test_control_duration.pdf")
