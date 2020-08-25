import pandas as pd
import numpy as np

# Age groups in polymod study
age_groups = ["00-04", "05-09", "10-14", "15-19", "20-24",
              "25-29", "30-34", "35-39", "40-44", "45-49",
              "50-54", "55-59", "60-64", "65-69", "70+"]

# Original colour scheme
colours = ["#43AA48", "#005838", "#008444", "#8DC63F", "#D7DF23",
           "#465EAB", "#262262", "#2B3990", "#27AAE1",
           "#357EC1", "#7663AB", "#472F8B",
           "#F04C31", "#873220", "#BD3F26"]

data_folder = "data/output_data/"
no_fatigue_filepath = data_folder + "sim_data_no_fatigue.csv"
with_fatigue_filepath = data_folder + "sim_data_with_fatigue.csv"
gradual_relaxation_filepath = data_folder + "sim_data_gradual_relaxation.csv"
exp_design_filepath = data_folder + "exp_design_matrix.csv"

cm_filepath = data_folder + "uk_baseline_contact_matrix_corrected.csv"
uk_pop_filepath = data_folder + "UK_population_by_polymod_categories.csv"
uk_pop_verity_filepath = data_folder + "UK_population_by_verity_categories.csv"

def get_TNG(cm, beta, g, S, N):
    """
    Calculate leading eigenvalue and associated eigenvector from
    next-generation matrix
    :param cm: Contact matrix
    :param beta: transmissibility vector
    :param g: infectious period
    :param S: Suscpetible population vector
    :param N: Total population vector
    :return: leading eigenvalue and associated eigenvector
    """

    n_ages = S.shape[0]

    # derivative of force of infection w.r.t I
    F = np.array([[beta[i]*cm[i, j]*S[i]/N[j] for j in range(n_ages)]
                  for i in range(n_ages)])

    Vinv = (1./g)*np.identity(n_ages, dtype=float)
    G = F.dot(Vinv)
    eigen = np.linalg.eig(G)
    lev_index = eigen[0].argmax()  # leading eigenvalue index
    Reff = eigen[0][lev_index]
    leading_ev = eigen[1][lev_index]
    return Reff, leading_ev


def get_params(R0=2.3):

    population = pd.read_csv(uk_pop_filepath,
                             index_col=0)
    contact_matrix = pd.read_csv(cm_filepath, index_col=0)

    # Number of age groups equal to contact matrix
    na = contact_matrix.shape[0]

    ag = contact_matrix.index.values

    par = {"g": (1. / 3.), "rho": 1. / 3, "eta": np.array([0.] * na),
           "n_ages": na, "tmax": 100 + 365, "tint": 57, "tint2": 57+100,
           "dt": 1, "I_stages": 4, "E_stages": 4,
           "age_groups": ag, "N": population["population"].values,
           "cm": contact_matrix.values.copy(),
           "stay_duration": 7, #12.8,
           "I_stages": 4,
           "E_stages": 4
           }
    beta_temp = np.ones(na)
    # beta_temp[0:4] = 0.5  # 50% susceptibility for under

    # Value of R0 assuming beta = beta_temp for each age class
    const = get_TNG(cm=par["cm"], beta=beta_temp, g=par["g"],
                    S=par["N"], N=par["N"])[0]

    # Set beta to get desired R0 (assume fixed constant for each age class)
    par["beta"] = (R0 / const) * beta_temp
    par["R0_baseline"] = R0

    return par


def get_social_distance_strats():

    n_strats = 6
    int_strats = pd.DataFrame({"name": "Self-isolation (SI)",
                               "q_y": 0.0, "q_a": 0.0, "q_o": 0.0,
                               "q_ya": 0.0,  "q_ao": 0.0, "q_yo": 0.0,
                               "color": "#8C9EDE"
                               }, index=np.arange(0, n_strats))

    int_strats.iloc[1, :] = pd.Series({"name": "SI and 60+ social"
                                               " distancing (SD)",
                                       "q_y": 0.0, "q_a": 0.0, "q_o": 0.5,
                                       "q_ya": 0.0,  "q_ao": 0.7, "q_yo": 0.9,
                                       "color": "#65C8D0"
                                       })
    int_strats.iloc[2, :] = pd.Series({"name": "SI; 25-59 and 60+ SD",
                                       "q_y": 0.0, "q_a": 0.5, "q_o": 0.5,
                                       "q_ya": 0.0,  "q_ao": 0.7, "q_yo": 0.9,
                                       "color": "#00767B"
                                       })
    int_strats.iloc[3, :] = pd.Series({"name": "SI; 0-24 and 60+ SD",
                                       "q_y": 0.7, "q_a": 0.0, "q_o": 0.5,
                                       "q_ya": 0.2,  "q_ao": 0.7, "q_yo": 0.9,
                                       "color": "#285C9F"
                                       })
    int_strats.iloc[4, :] = pd.Series({"name": "SI; 0-24, 25-59 and 60+ SD",
                                       "q_y": 0.7, "q_a": 0.5, "q_o": 0.5,
                                       "q_ya": 0.2,  "q_ao": 0.7, "q_yo": 0.9,
                                       "color": "#04375E"
                                       })

    int_strats.iloc[5, :] = pd.Series({"name": "SI and no 60+ contacts",
                                       "q_y": 0.0, "q_a": 0.0, "q_o": 1.0,
                                       "q_ya": 0.0,  "q_ao": 1.0, "q_yo": 1.0,
                                       "color": "#770000"
                                       })
    int_strats.index.name = "Intervention strategy"

    return int_strats


def get_self_isolation_parameters():
    return {"prob_s": [1], "comp_factor": np.arange(0, 1.01, 0.01)}


def get_social_dist_matrix(par, var_par):

    vp = var_par  # to shorten code
    n_ages = par["n_ages"]

    mat = np.array([[vp["q_y"], vp["q_ya"], vp["q_yo"]],
                    [vp["q_ya"], vp["q_a"], vp["q_ao"]],
                    [vp["q_yo"], vp["q_ao"], vp["q_o"]]])

    sd_mat = pd.DataFrame(np.ones([n_ages, n_ages]), dtype="float")

    # Divide ages groups into young, adult and older respectively:
    # Note: this is not robust to repartitioning of age groups
    age_types = [slice(0, 5), slice(5, 12), slice(12, n_ages)]

    for i in range(len(age_types)):
        for j in range(len(age_types)):
            sd_mat.iloc[age_types[i], age_types[j]] = mat[i, j]

    return sd_mat


def get_self_isolation_factor(var_par):
    return var_par["prob_s"] * (1 - var_par["comp_factor"])\
           + (1 - var_par["prob_s"])


# multipliers matrix
def get_intervention_R0(par, var_par, sd_on=False):

    sd = get_social_dist_matrix(par=par, var_par=var_par)
    self_isolation = get_self_isolation_factor(var_par=var_par)

    cm_int = self_isolation * (par["cm"] * (1 - sd_on * sd)).values

    next_gen_mat = get_TNG(cm=cm_int, beta=par["beta"],
                           g=par["g"], S=par["N"], N=par["N"])

    return next_gen_mat[0]


def get_hosp_rate():

    # read in hospitalisation rate data
    h_df = pd.read_csv(
        "data/Verity_data/Table 3-Estimates of hospitalisation.csv",
        index_col=0)

    # read in population data using Verity et al bins
    pop_df = pd.read_csv(uk_pop_verity_filepath, sep=",")
    pop_df.index = h_df.index
    pop_vec = pop_df["population"]

    # Convert from percentage to fraction
    h_df = h_df.iloc[:, 2]/100
    # Calculate 70+ hospitalisations from 70-79 and 80+ entries
    h_df["70+"] = (h_df*pop_vec)[["70–79", "≥80"]].sum()\
        / pop_vec[["70–79", "≥80"]].sum()

    # Duplicate first 7 categories to convert from 10 yr increments to 5 yr
    h_rate_study = h_df.iloc[np.arange(0, 7).repeat(2)]
    h_rate_study["70+"] = h_df["70+"]
    h_rate_study.index = age_groups
    return h_rate_study


def get_fatality_rate(df, n_ages=15, stay_duration=20):

    # read in fatality rate data
    inf_df = pd.read_csv("./data/Verity_data/Table 1-Estimates of case "
                         "fatality ratio.csv", sep=",")
    inf_df.index = inf_df["Subset"]

    # read in pop data using Verity et al bins
    pop_df = pd.read_csv(uk_pop_verity_filepath, sep=",")
    pop_df.index = inf_df.iloc[1:-2].index
    pop_vec = pop_df["population"]

    inf_rate = inf_df.loc[:, "Infection fatality ratio"]/100
    inf_rate["70+"] = (inf_rate*pop_vec)[["70–79", "≥80"]].sum()\
        / pop_vec[["70–79", "≥80"]].sum()

    # get rates for columns used in study
    inf_rate_study = inf_rate.iloc[np.arange(1, 8).repeat(2)]
    inf_rate_study["70+"] = inf_rate["70+"]
    inf_rate_study.index = age_groups

    # cases columns
    h_df = get_hosp_rate()
    C_cols = ["C" + str(y) for y in range(0, n_ages)]
    h_df.index = C_cols

    inf_df = inf_rate_study.copy()
    C_cols = ["C" + str(y) for y in range(0, n_ages)]
    inf_df.index = C_cols

    # caclulate daily fatalities for under 60 and over 60
    inf_df_l60 = inf_df.iloc[:12]
    inf_df_g60 = inf_df.iloc[12:]

    df["<60_fatal"] = df[C_cols[:12]].apply(lambda x: x.dot(inf_df_l60), axis=1)
    df[">60_fatal"] = df[C_cols[12:]].apply(lambda x: x.dot(inf_df_g60), axis=1)

    # calculate daily hospitalisations
    df["new_hosp"] = df[C_cols].apply(lambda x: x.dot(h_df), axis=1)

    # calculate total hospital burden
    df["hosp_burden"] = df.groupby("run")["new_hosp"].diff(
        periods=stay_duration)


def get_two_type_params(index_map, R0=2.3):

    def get_transformed(cm, N, index_map):
        N_new = np.array([N[A].sum() for A in index_map])
        cm_scaled = np.array([[N[i] * cm[i, j] for j in range(cm.shape[0])]
                              for i in range(cm.shape[1])])
        cm_new = np.array([[cm_scaled[np.ix_(A, B)].sum() / N_new[a]
                            for b, B in enumerate(index_map)]
                           for a, A in enumerate(index_map)])
        return cm_new, N_new


    par = get_params(R0=R0)

    par["cm"], par["N"] = get_transformed(par["cm"], par["N"], index_map)

    na = len(index_map)
    par["n_ages"] = na
    par["eta"] = np.array([0.] * na)

    # Value of R0 assuming beta = 1 for each age class
    const = get_TNG(cm=par["cm"], beta=np.ones(na), g=par["g"],
                    S=par["N"], N=par["N"])[0]

    # Set beta to get desired R0 (assume fixed constant for each age class)
    par["beta"] = (R0 / const) * np.ones(na)
    par["R0_baseline"] = R0
    # Hospitalisation rate in the two type model
    # (from age-structured SEIR model; summary_figure.py))
    par["hosp_rate"] = 0.03680

    return par


def convert_to_data_frame(output, par):
    """
    Convert simulation output to data frame
    :param output: output of ode solver
    :param par: simulation parameters
    :return: dataframe of simulation output
    """
    n_ages = par["n_ages"]
    I_stages = par["I_stages"]
    E_stages = par["E_stages"]

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


# Initial conditions
def set_initial_conditions(par):
    """
    Set SEIR model initial conditions assuming one individual initially infected.
    :param par:
    :return:
    """
    n_ages = par["n_ages"]
    I_stages = par["I_stages"]
    E_stages = par["E_stages"]
    I = np.zeros((I_stages, n_ages), dtype="float")
    I[0, 0] = 1
    E = np.zeros((E_stages, n_ages), dtype="float")
    R = np.zeros(n_ages, dtype="float")
    C = np.zeros(n_ages, dtype="float")
    S = par["N"] - I.sum(axis=0)
    initial_cond = np.concatenate((S, E.flatten(), I.flatten(), R, C))
    return initial_cond


# Split compartments from single vector
def split_seir_compartments(y, par):
    """
    Split compartments from single long vector into in tuple of model
    compartments based on infection status
    :param y:
    :param par:
    :return:
    """
    n_ages = par["n_ages"]
    I_stages = par["I_stages"]
    E_stages = par["E_stages"]

    S = y[0:n_ages]
    E = y[n_ages:((1 + E_stages) * n_ages)].reshape(E_stages, -1)
    I = y[((1 + E_stages) * n_ages):((1 + E_stages + I_stages) * n_ages)] \
        .reshape(I_stages, -1)
    R = y[((1 + E_stages + I_stages) * n_ages):(
            (2 + E_stages + I_stages) * n_ages)]
    C = y[((2 + E_stages + I_stages) * n_ages):(
            (3 + E_stages + I_stages) * n_ages)]

    return S, E, I, R, C

# Get contact matrix for given intervention strengths
def get_intervention_cm(par, var_par):

    sd = get_social_dist_matrix(par=par, var_par=var_par)
    self_isolation = get_self_isolation_factor(var_par=var_par)
    cm_int = self_isolation*(par["cm"]*(1-sd)).values
    return cm_int



def age_structured_SEIR_model(S, E, I, R, C, par):

    beta = par["beta"]
    rho = par["rho"]
    g = par["g"]
    eta = par["eta"]
    cm = par["cm_sd"] # todo: fix code
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
