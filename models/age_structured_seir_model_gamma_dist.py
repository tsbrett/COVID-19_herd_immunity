import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import models.helper as h
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

n_cores = 7  # todo: adjust to make actual number of cores

def solve_SIRS_model(par, var_par, include_fatigue=True):

    def model_wrapper(t, y, par):

        # Split compartments
        S, E, I, R, C = h.split_seir_compartments(y=y, par=par)
        # Run simulation
        S, E, I, R, C = h.age_structured_SEIR_model(S=S, E=E, I=I, R=R,
                                                    C=C, par=par)
        # Combine compartments
        y_new = np.concatenate((S, E.flatten(), I.flatten(), R, C))

        return y_new

    # Start of interventions

    # End all interventions not affecting older age classes
    var_par_off = var_par.copy()
    if include_fatigue:
        var_par_off[["q_a", "q_ya", "q_y"]] = 0

    cm_no_int = par["cm"]
    cm_int1 = h.get_intervention_cm(par=par, var_par=var_par)
    cm_int2 = h.get_intervention_cm(par=par, var_par=var_par_off)


    # Initial conditions
    initial_cond = h.set_initial_conditions(par)

    # Crude implementation of applying controls (off->on->reduced)
    par["cm_sd"] = cm_no_int
    sol = solve_ivp(fun=model_wrapper, t_span=[0, par["tint"]],
                    y0=initial_cond, method="RK45",
                    t_eval=np.arange(0, par["tint"]+1, 1),
                    args=(par,))

    par["cm_sd"] = cm_int1
    sol2 = solve_ivp(fun=model_wrapper, t_span=[par["tint"], par["tint2"]],
                     y0=sol["y"][:, -1], method="RK45",
                     t_eval=np.arange(par["tint"]+1, par["tint2"]+1, 1),
                     args=(par,))

    par["cm_sd"] = cm_int2
    sol3 = solve_ivp(fun=model_wrapper, t_span=[par["tint2"], par["tmax"]],
                     y0=sol2["y"][:, -1], method="RK45",
                     t_eval=np.arange(par["tint2"]+1, par["tmax"]+1, 1),
                     args=(par,))

    # # Reset par["cm"] to its original value
    # par["cm"] = cm_no_int

    # Convert output to dataframe
    ret_df = pd.concat([h.convert_to_data_frame(output=sol, par=par),
                        h.convert_to_data_frame(output=sol2, par=par),
                        h.convert_to_data_frame(output=sol3, par=par)], axis=0)
    return ret_df


def out_df_wrapper(i, par, var_par, include_fatigue=True):
    n_ages = par["n_ages"]

    out_df = solve_SIRS_model(par=par, var_par=var_par,
                              include_fatigue=include_fatigue)
    out_df["run"] = i
    I_cols = ["I" + str(y) for y in range(0, n_ages)] # todo: why is this done here?
    out_df["I_tot"] = out_df[I_cols].sum(axis=1)
    return out_df

#%%
# todo check if i need to tweak start date for control with different R0


def run_seir_simulations(par, include_fatigue=True):



    if include_fatigue:
        filepath = h.with_fatigue_filepath
    else:
        filepath = h.no_fatigue_filepath

    exp_design = pd.read_csv(h.exp_design_filepath, index_col=0)

    start_time = time.time()
    out = Parallel(n_jobs=n_cores)(delayed(out_df_wrapper)(i, par=par,
                                                           var_par=r,
                                                           include_fatigue=include_fatigue)
                                   for i, r in exp_design.iterrows())
    print("--- %s seconds ---" % (time.time() - start_time))

    out = pd.concat(out)
    out = out.sort_values(["run", "time"])

    h.get_fatality_rate(out, stay_duration=par["stay_duration"])
    print("--- %s seconds ---" % (time.time() - start_time))

    out.to_csv(filepath)


def single_seir_run_test(par):

    exp_design = pd.read_csv(h.exp_design_filepath, index_col=0)
    out_df = solve_SIRS_model(par=par, var_par=exp_design.iloc[0])
    out_df["run"] = 0
    I_cols = ["I" + str(y) for y in range(0, par["n_ages"])]
    out_df["I_tot"] = out_df[I_cols].sum(axis=1)

    return out_df


def get_last_day_below(par, threshold=10000):

    df = single_seir_run_test(par=par)
    I_tot_max = df["I_tot"].cummax()

    return I_tot_max[(I_tot_max < threshold)].index.max()


def single_run_test_plot(par, t2=2):

    df = single_seir_run_test(par=par)
    fig, axes = plt.subplots(figsize=(5, 4))
    ax = axes
    ax.plot(np.log10(df["I_tot"]))
    x = np.arange(0,50,2)
    ax.plot(np.log10(df["I_tot"]))
    ax.plot(x, x*np.log10(2)/t2)

    return fig

#fig_single = single_run_test_plot(R0=6.0)
#plt.show()


