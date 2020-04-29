import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import models.helper as h
import time


def model(t, S, E, I, R, C, par, sd_func):

    beta = par["beta"]
    rho = par["rho"]
    g = par["g"]
    eta = par["eta"]
    cm = par["cm"]
    N = par["N"] # assume constant age profile
    n_ages = len(S) # note this is not robust
    E_stages = par["E_stages"]
    I_stages = par["I_stages"]

    I_sum = I.sum(axis=0) # sum over stages


    # Social distancing at time t
    sd = sd_func(t)

    # Force of infection
    foi = np.array([beta[i]*np.sum([sd[i,j]*I_sum[j]/N[j]
                                    for j in range(n_ages)])
                    + eta[i]/N[i] for i in range(n_ages)])


    I_shifted = np.roll(I, 1, axis=0)
    I_shifted[0, :] = 0
    E_shifted = np.roll(E, 1, axis=0)
    E_shifted[0, :] = 0

    dS = - foi*S
    dE = - (rho*E_stages)*E + (rho*E_stages)*E_shifted
    dE[0, :] += foi*S
    dI = - (g*I_stages)*I + (g*I_stages)*I_shifted
    dI[0, :] += rho*E_stages*E[-1, :]
    dR = g*I_stages*I[-1, :]
    dC = dR

    return (dS, dE, dI, dR, dC)


def solve_SIRS_model(par, var_par, include_fatigue=True, use_pulsed=False):

    n_ages = par["n_ages"]
    I_stages = par["I_stages"]
    E_stages = par["E_stages"]

    # Specify initial conditions
    I = np.zeros((I_stages, n_ages), dtype="float")
    I[0,0] = 1
    E = np.zeros((E_stages, n_ages), dtype="float")
    R = np.zeros(n_ages, dtype="float")
    C = np.zeros(n_ages, dtype="float")
    S = par["N"] - I.sum(axis=0)
    initial_cond = np.concatenate((S, E.flatten(), I.flatten(), R, C))

    # Start of interventions
    sd = h.get_social_dist_matrix(par=par, var_par=var_par)
    self_isolation = h.get_self_isolation_factor(var_par=var_par)

    if include_fatigue:
        var_par_off = var_par.copy()
        var_par_off[["q_a", "q_ya", "q_y"]] = 0
        sd2 = h.get_social_dist_matrix(par=par, var_par=var_par_off)
    else:
        sd2 = h.get_social_dist_matrix(par=par, var_par=var_par)

    cm_sd_on = self_isolation*(par["cm"]*(1-sd)).values
    cm_sd_off = self_isolation*(par["cm"]*(1-sd2)).values

    dc = par["dc"]

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

    def split_compartments(y):

        S = y[0:n_ages]
        E = y[n_ages:((1 + E_stages) * n_ages)].reshape(E_stages, -1)
        I = y[
            ((1 + E_stages) * n_ages):((1 + E_stages + I_stages) * n_ages)].reshape(
            I_stages, -1)
        R = y[((1 + E_stages + I_stages) * n_ages):(
                (2 + E_stages + I_stages) * n_ages)]
        C = y[((2 + E_stages + I_stages) * n_ages):(
                    (3 + E_stages + I_stages) * n_ages)]

        return S, E, I, R, C

    def combine_compartments(S, E, I, R, C):
        return np.concatenate((S, E.flatten(), I.flatten(), R, C))

    def model_wrapper(t, y, par):

        S, E, I, R, C = split_compartments(y)
        S, E, I, R, C = model(t=t, S=S, E=E, I=I, R=R, C=C, par=par,
                              sd_func=get_social_distancing)
        return combine_compartments(S, E, I, R, C)

    sol = solve_ivp(fun=model_wrapper, t_span=[0, par["tmax"]],
                    y0=initial_cond, method="RK45",
                    t_eval=np.arange(0, par["tmax"]+1, 1),
                    args=(par,))

    def get_data_frame(output):

        y = output["y"]
        S = y[0:n_ages]
        R = y[((1 + E_stages + I_stages) * n_ages):((2 + E_stages + I_stages)
                                                    * n_ages)]
        C = y[((2 + E_stages + I_stages) * n_ages):((3 + E_stages + I_stages)
                                                    * n_ages)]

        E = y[n_ages:((1 + E_stages) * n_ages)]

        I = y[((1 + E_stages) * n_ages):((1 + E_stages + I_stages) * n_ages)]

        I_sum = np.array([I[i::n_ages, :].sum(axis=0) for i in range(n_ages)])
        E_sum = np.array([E[i::n_ages, :].sum(axis=0) for i in range(n_ages)])

        y = np.concatenate((S, E_sum, I_sum, R, C), axis=0)

        df = pd.DataFrame(y.transpose())
        df.columns = [x + str(y) for x in ["S", "E", "I", "R", "C"] for y in
                      range(0, n_ages)]
        df.index = output["t"]
        df.index.name = "time"
        return df

    out_df = get_data_frame(sol)

    return out_df


#%%

n_ages = 15
stay_duration = 12.8
params = h.get_params(None)
params["I_stages"] = 4
params["E_stages"] = 4


exp_design = pd.read_csv(h.exp_design_filepath, index_col=0)


#%%

def out_df_wrapper(dc, par, var_par, include_fatigue=True, use_pulsed=False):
    par["dc"] = dc
    par["tint2"] = 57

    out_df = solve_SIRS_model(par=par, var_par=var_par,
                              include_fatigue=include_fatigue,
                              use_pulsed=use_pulsed)
    out_df["run"] = dc
    I_cols = ["I" + str(y) for y in range(0, n_ages)]
    out_df["I_tot"] = out_df[I_cols].sum(axis=1)
    return out_df


#%% Runs with fatigue

exp_design_filter = exp_design.iloc[54]
# params["tint"] = 65
#
params["t_on"] = 80
#
params["t_off"] = 30
params["tscale"] = 1000
params["eta"] = 1./15*np.ones(n_ages)

dc_list = [1, 100, 200, 300, ]


start_time = time.time()

out = [out_df_wrapper(dc, par=params, var_par=exp_design_filter,
                      include_fatigue=True)
       for dc in dc_list]

print("--- %s seconds ---" % (time.time() - start_time))

out_no_int = out_df_wrapper(10, par=params, var_par=exp_design.iloc[0],
                            include_fatigue=True)
out_no_int["run"] = 0

out = pd.concat([out_no_int] + out)

h.get_fatality_rate(out, stay_duration=stay_duration)

out.to_csv(h.gradual_relaxation_filepath)
