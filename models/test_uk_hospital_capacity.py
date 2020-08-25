import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates



df = pd.read_csv("./data/UK_patients_in_hospital_data.csv")
df2 = pd.read_csv("./data/2020-06-26_COVID-19_Data/People in Hospital (UK)-Table 1.csv",
                  skiprows=3, header=1, usecols=np.arange(0,11),
                  skipfooter=12, engine="python")



df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")


df2["Date"] = pd.to_datetime(df2["Date"], format="%d/%m/%Y")
df2.index = df2["Date"]
df2["All UK"] = df2.sum(axis=1)

#%%
# plt.plot(df["date"], df["hospitalCases"])
label_fontsize = 10
fontsize = 8
date_fmt = mdates.DateFormatter('%m-%d')

april_mean = df2[df2["Date"].dt.month == 4].mean()

fig, axes = plt.subplots(nrows=2, figsize=(5, 6))
ax = axes[0]
for c in df2.columns[1:-1]:
    ax.plot(df2[c], label=c)
ax.set_ylim(0, 6000)

ax = axes[1]
ax.plot(df2["All UK"], label="All UK")
ax.axhline(april_mean["All UK"],
           xmin=0.16, xmax=0.43, linestyle="--", color="grey")
# ax.axvline(pd.to_datetime("2020-04-01"))
# ax.axvline(pd.to_datetime("2020-04-30"))


titles = ["a)", "b)", "c)", "d)", "e)"]
for i, ax in enumerate(axes):
    ax.legend(frameon=False, fontsize=fontsize, ncol=2)
    ax.set_title(titles[i], loc="left", fontsize=label_fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=1, interval=2))
    sns.despine(ax=ax, offset=5, trim=True)


fig.tight_layout()
fig.savefig("./figures/fig_test_uk_hosp.pdf")
plt.show()
print(df2[df2["Date"].dt.month == 4].mean())

