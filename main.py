import models.helper as h
import models.polymod_matrix as pm
import models.create_exp_design_matrix as cd
import models.age_structured_seir_model_gamma_dist as sim
import models.age_structured_seir_model_gamma_dist_varying_control as sim_vc
import models.incidence_plot as intro
import models.suppression_analysis as sa
import models.mitigation_analysis as ma
import models.gradual_relaxation_analysis as gr
import models.age_structured_seir_model_two_type as tt
import models.summary_figure as sf


# Generate contact matrix and age-specifc population sizes for model
pm.create_polymod_matrix()

# Get model parameters
params = h.get_params(R0=2.3)  # Uses generated contact matrix

# Generate intervention parameter sets for simulations
cd.create_exp_design_matrix(par=params)

# Set intervention start and end dates relative to last day I < threshold
ldb = sim.get_last_day_below(par=params, threshold=10000)
params["tint"] = ldb + 1
params["tint2"] = params["tint"] + 100

#%%
# Run simulations (with and without fatigue)
sim.run_seir_simulations(par=params, include_fatigue=True)
sim.run_seir_simulations(par=params, include_fatigue=False)

#%%
# Run simulations with gradual relaxation of control measures
sim_vc.run_seir_varying_control_simulations(par=params)

#%%
# Plot examples of time series and interventions (Fig 2)
intro.plot_introduction_figure(par=params)

# Application to disease suppression (Fig 3)
sa.plot_suppression_results(par=params)

# Application to disease mitigation (Fig 4)

# Calculate mitigation summary stats (with and without fatigue)
ts_max = ma.construct_ts_max(h.with_fatigue_filepath, par=params)
ts_max_nf = ma.construct_ts_max(h.no_fatigue_filepath, par=params)

# Print mitigation Reff ranges
ma.print_mitigation_ranges(ts_max_nf, hosp_cap=1e5)

# Generate plot
ma.plot_mitigation_summary(ts_max_nf, ts_max)

#%%
# Hospital burden analysis (Fig 5)
gr.plot_gradual_relaxation(par=params)

# Generate summary figure (Fig 6)
sf.plot_summary_figure(par=params)

#%%
# Simulate and plot two type seir model (WIP)
R0 = params["R0_baseline"]
out_df_two, out_stats = tt.run_seir_varying_control_simulations(var="hosp_cap",
                                                                R0=R0)
tt.plot_optimal_strategy(out_df=out_df_two, out_stats=out_stats)

params_two_type, df_tt = tt.get_heatmap_data(R0=R0)

tt.plot_control_duration_heatmap(par=params_two_type, df=df_tt)
