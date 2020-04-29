# Bad way of running analysis but it works

# Generate contact matrix and age-specifc population sizes for model
import models.polymod_matrix

# Generate intervention parameter sets for simulations
import models.create_exp_design_matrix

# Run simulations
import models.age_structured_seir_model_gamma_dist

# Run simulations with gradual relaxation of control measures
import models.age_structured_seir_model_gamma_dist_varying_control

# Examples of time series and interventions (Fig 2)
import models.incidence_plot

# Application to disease suppression (Fig 3)
import models.suppression_analysis

# Application to disease migigation (Fig 4)
import models.mitigation_analysis

# Hospital burden analysis (Fig 5)
import models.gradual_relaxation_analysis

# Generate summary figure (Fig 6)
import models.summary_figure
