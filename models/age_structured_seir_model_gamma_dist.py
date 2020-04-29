import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import models.helper as h
import time

def model(S, E, I, R, C, par):

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

    # Force of infection
    foi = np.array([beta[i]*np.sum([cm[i, j]*I_sum[j]/N[j]
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


def solve_SIRS_model(par, var_par, include_fatigue=True):
    n_ages = par["n_ages"]
    I_stages = par["I_stages"]
    E_stages = par["E_stages"]

    # Initial conditions
    I = np.zeros((I_stages, n_ages), dtype="float")
    I[0, 0] = 1
    E = np.zeros((E_stages, n_ages), dtype="float")
    R = np.zeros(n_ages, dtype="float")
    C = np.zeros(n_ages, dtype="float")
    S = par["N"] - I.sum(axis=0)
    initial_cond = np.concatenate((S, E.flatten(), I.flatten(), R, C))

    # Start of interventions
    sd = h.get_social_dist_matrix(par=par, var_par=var_par)
    self_isolation = h.get_self_isolation_factor(var_par=var_par)
    cm_int1 = self_isolation*(par["cm"]*(1-sd)).values

    # End all interventions not affection older age classes
    var_par_off = var_par.copy()

    if include_fatigue:
        var_par_off[["q_a", "q_ya", "q_y"]] = 0

    sd = h.get_social_dist_matrix(par=par, var_par=var_par_off)
    # self_isolation = get_self_isolation_factor(var_par=var_par_off)
    cm_int2 = self_isolation*(par["cm"]*(1-sd)).values

    def split_compartments(y):

        S = y[0:n_ages]
        E = y[n_ages:((1 + E_stages) * n_ages)].reshape(E_stages, -1)
        I = y[((1 + E_stages) * n_ages):((1 + E_stages + I_stages) * n_ages)]\
            .reshape(I_stages, -1)
        R = y[((1 + E_stages + I_stages) * n_ages):(
                (2 + E_stages + I_stages) * n_ages)]
        C = y[((2 + E_stages + I_stages) * n_ages):(
                    (3 + E_stages + I_stages) * n_ages)]

        return S, E, I, R, C

    def combine_compartments(S, E, I, R, C):
        return np.concatenate((S, E.flatten(), I.flatten(), R, C))

    def model_wrapper(t, y, par):

        S, E, I, R, C = split_compartments(y)
        S, E, I, R, C = model(S=S, E=E, I=I, R=R, C=C, par=par)

        return combine_compartments(S, E, I, R, C)

    # Crude implementation of applying controls (off->on->reduced)
    cm_og = par["cm"]

    sol = solve_ivp(fun=model_wrapper, t_span=[0, par["tint"]],
                    y0=initial_cond, method="RK45",
                    t_eval=np.arange(0, par["tint"]+1, 1),
                    args=(par,))


    par["cm"] = cm_int1
    sol2 = solve_ivp(fun=model_wrapper, t_span=[par["tint"],par["tint2"]],
                     y0=sol["y"][:,-1], method="RK45",
                     t_eval=np.arange(par["tint"]+1,par["tint2"]+1,1),
                     args=(par,))

    par["cm"] = cm_int2
    sol3 = solve_ivp(fun=model_wrapper, t_span=[par["tint2"], par["tmax"]],
                     y0=sol2["y"][:, -1], method="RK45",
                     t_eval=np.arange(par["tint2"]+1, par["tmax"]+1, 1),
                     args=(par,))

    # Reset par["cm"] to its original value
    par["cm"] = cm_og

    def get_data_frame(output):

        # get output I
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

    ret_df = pd.concat([get_data_frame(sol), get_data_frame(sol2),
                        get_data_frame(sol3)], axis=0)
    return ret_df


#%%

stay_duration = 12.8
params = h.get_params(None)
n_ages = params["n_ages"]
params["I_stages"] = 4
params["E_stages"] = 4

exp_design = pd.read_csv(h.exp_design_filepath, index_col=0)



#%%

# test run
out_df = solve_SIRS_model(par=params, var_par=exp_design.iloc[0])
out_df["run"] = 0


#%%
def out_df_wrapper(i, par, var_par, include_fatigue=True):
    out_df = solve_SIRS_model(par=par, var_par=var_par,
                              include_fatigue=include_fatigue)
    out_df["run"] = i
    I_cols = ["I" + str(y) for y in range(0, n_ages)]
    out_df["I_tot"] = out_df[I_cols].sum(axis=1)
    return out_df


#%% Runs with fatigue
exp_design_filter = exp_design.iloc[0:]

start_time = time.time()
out = pd.concat([out_df_wrapper(i, par=params, var_par=r, include_fatigue=True)
                 for i, r in exp_design_filter.iterrows()
                 ])
h.get_fatality_rate(out, stay_duration=stay_duration)
print("--- %s seconds ---" % (time.time() - start_time))

out.to_csv(h.with_fatigue_filepath)


#%% Runs without fatigue
exp_design_filter = exp_design.iloc[0:]

start_time = time.time()
out = pd.concat([out_df_wrapper(i, par=params, var_par=r, include_fatigue=False)
                 for i, r in exp_design_filter.iterrows()
                 ])
h.get_fatality_rate(out, stay_duration=stay_duration)
print("--- %s seconds ---" % (time.time() - start_time))

out.to_csv(h.no_fatigue_filepath)



