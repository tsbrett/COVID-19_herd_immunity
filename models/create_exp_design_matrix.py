import pandas as pd
import models.helper as h
from itertools import product

params = h.get_params(None)


def get_r0_wrapper(x):
    return h.get_intervention_R0(par=params, var_par=x, sd_on=True)


# Dict of self-isolation probability ("prob_s") and engagement ("comp_factor")
# levels
si_par = h.get_self_isolation_parameters()
# Data frame of social distancing strategies
sd_strats = h.get_social_distance_strats()
n_strats = sd_strats.shape[0]

# Convert SI levels to data frame
si_df = pd.DataFrame(list(product(*list(si_par.values()))),
                     columns=list(si_par.keys()))
# Replicate by the number of SD strategies
si_df_rep = si_df.loc[si_df.index.repeat(n_strats)]

# Replicate by the number of SI levels
sd_stats_rep = pd.concat([sd_strats] * si_df.shape[0])

# Combine SI levels and SD strats to get full design matrix
exp_design = pd.concat([si_df_rep.reset_index(drop=True),
                        sd_stats_rep.reset_index()],
                       axis=1)

# Calculate R0 under intervention and SI effectiveness
exp_design["R0"] = exp_design.apply(get_r0_wrapper, axis=1)
exp_design["si_eff"] = 1-exp_design.apply(h.get_self_isolation_factor, axis=1)

exp_design.to_csv(h.exp_design_filepath)
