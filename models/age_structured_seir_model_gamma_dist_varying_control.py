import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import models.helper as h
import time


def solve_SIRS_model(par, var_par, include_fatigue=True, use_pulsed=False):

    dc = par["dc"]

    initial_cond = h.set_initial_conditions(par)

    # Start of interventions
    sd = h.get_social_dist_matrix(par=par, var_par=var_par)
    self_isolation = h.get_self_isolation_factor(var_par=var_par)

    var_par_off = var_par.copy()
    if include_fatigue:
        var_par_off[["q_a", "q_ya", "q_y"]] = 0

    cm_sd_on = h.get_intervention_cm(par=par, var_par=var_par)
    cm_sd_off = h.get_intervention_cm(par=par, var_par=var_par_off)

    # function controlling social distancing protocol
    def get_social_distancing_no_pulse(t):
        if t < par["tint"]:
            return par["cm"]

        elif t < par["tint2"]:
            return self_isolation*(par["cm"]*(1-sd)).values

        elif t < par["tint2"] + dc:

            var_par_off[["q_a", "q_ya", "q_y"]] = (1 - (t-par["tint2"]) / dc)\
                                                  * var_par[["q_a", "q_ya",
                                                             "q_y"]]
            sd2 = h.get_social_dist_matrix(par=par, var_par=var_par_off)
            return self_isolation * (par["cm"] * (1 - sd2)).values

        else:
            var_par_off[["q_a", "q_ya", "q_y"]] = 0
            sd2 = h.get_social_dist_matrix(par=par, var_par=var_par_off)

            return self_isolation * (par["cm"] * (1 - sd2)).values

    # function controlling social distancing protocol (pulsed)
    def get_social_distancing_pulsed(t):

        tp = t - par["tint"]
        td = par["t_on"] + par["t_off"]
        if tp < 0:
            return par["cm"]

        elif ((tp / td) % 1) < ((1 - (t / par["tscale"])**2)
                                * par["t_on"] / td):

            return cm_sd_on

        else:
            return cm_sd_off

    if use_pulsed:
        get_social_distancing = get_social_distancing_pulsed
    else:
        get_social_distancing = get_social_distancing_no_pulse

    def model_wrapper(t, y, par):

        S, E, I, R, C = h.split_seir_compartments(y, par=par)
        par["cm_sd"] = get_social_distancing(t)

        S, E, I, R, C = h.age_structured_SEIR_model(S=S, E=E, I=I, R=R,
                                                    C=C, par=par)

        y_new = np.concatenate((S, E.flatten(), I.flatten(), R, C))
        return y_new

    sol = solve_ivp(fun=model_wrapper, t_span=[0, par["tmax"]],
                    y0=initial_cond, method="RK45",
                    t_eval=np.arange(0, par["tmax"] + 1, 1),
                    args=(par,))

    out_df = h.convert_to_data_frame(output=sol, par=par)

    return out_df


def out_df_wrapper(dc, par, var_par, include_fatigue=True, use_pulsed=False):
    par["dc"] = dc
    n_ages = par["n_ages"]
    out_df = solve_SIRS_model(par=par, var_par=var_par,
                              include_fatigue=include_fatigue,
                              use_pulsed=use_pulsed)
    out_df["run"] = dc
    I_cols = ["I" + str(y) for y in range(0, n_ages)]
    out_df["I_tot"] = out_df[I_cols].sum(axis=1)
    return out_df


def run_seir_varying_control_simulations(par):

    exp_design = pd.read_csv(h.exp_design_filepath, index_col=0)

    # todo: not robust to exp_design
    exp_design_filter = exp_design.iloc[11*6-2]

    # params["tint"] = 65
    # params["t_on"] = 80
    par["tint2"] = par["tint"]
    # params["t_off"] = 30
    par["tscale"] = 1000
    par["eta"] = 1./par["n_ages"]*np.ones(par["n_ages"])

    # Varying control runs
    dc_list = [1, 100, 200, 300, ]

    start_time = time.time()

    out = [out_df_wrapper(dc, par=par, var_par=exp_design_filter,
                          include_fatigue=True)
           for dc in dc_list]

    print("--- %s seconds ---" % (time.time() - start_time))

    out_no_int = out_df_wrapper(10, par=par, var_par=exp_design.iloc[0],
                                include_fatigue=True)
    out_no_int["run"] = 0

    out = pd.concat([out_no_int] + out)

    h.get_fatality_rate(out, stay_duration=par["stay_duration"])

    out.to_csv(h.gradual_relaxation_filepath)
