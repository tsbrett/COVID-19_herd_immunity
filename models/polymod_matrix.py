import numpy as np
import pandas as pd
import models.helper as h

# Read in original Mossong matrix for UK
cm = pd.read_csv("./data/MOSSONG-britain-per-day.csv", header=None)

age_groups = ["00-04", "05-09", "10-14", "15-19", "20-24",
              "25-29", "30-34", "35-39", "40-44", "45-49",
              "50-54", "55-59", "60-64", "65-69", "70+"]

cm.columns = age_groups
cm.index = age_groups

cm.index.name = "age of contact"
cm.columns.name = "age group of participant"

# Note the transpose w.r.t original Mossong matrix to match SEIR model notation
cm = cm.transpose()

# cm.to_csv("./data/output_data/mossong_britain_per_day_with_titles.csv")

#%%
# Get UK population by age categories of POLYMOD matrix
ages_df = pd.read_csv("./data/uk_demographic_data/2018-Table 1.csv",
                      index_col=0)
ages_df = ages_df.loc["UNITED KINGDOM", ]
ages_df = ages_df.iloc[1:]
ages_df = pd.DataFrame({"population": ages_df})
ages_df = ages_df.reset_index()

f_tot = ages_df.loc[(ages_df["index"] == "f_tot"), "population"].values
m_tot = ages_df.loc[(ages_df["index"] == "m_tot"), "population"].values

ages_df = ages_df.loc[~(ages_df["index"].isin(["m_tot", "f_tot"])), ]

ages_df[["sex", "year", "age"]] = ages_df.loc[:, "index"].str.split(pat="_",
                                                                    n=-1,
                                                                    expand=True)

ages_df["age_bins"] = pd.cut(ages_df["age"].astype("int"),
                             bins=np.concatenate([np.arange(0, 71, 5), [100]]),
                             right=False)

pop_by_age_bin = ages_df.groupby("age_bins")["population"].sum()
pop_by_age_bin.to_csv(h.uk_pop_filepath)


# Calculation of size of age categories for the bins used in Verity et al
ages_df["age_bins2"] = pd.cut(ages_df["age"].astype("int"),
                              bins=np.concatenate([np.arange(0, 81, 10),
                                                   [100]]),
                              right=False)

pop_by_age_bin2 = ages_df.groupby("age_bins2")["population"].sum()
pop_by_age_bin2.to_csv(h.uk_pop_verity_filepath)


#%%
# Correcting contact matrix for reciprocity

# total daily UK contacts between i and j as reported by participants
E1 = cm.multiply(pop_by_age_bin.values, axis=0)
# Symmetrise matrix
E2 = (E1 + E1.transpose())/2.
# Divide by population, c_ij = E2_ij/N_i to get corrected number of daily
# contacts for each participant
cm_corrected = E2.multiply(1/pop_by_age_bin.values, axis=0)

cm_corrected.to_csv(h.cm_filepath)
