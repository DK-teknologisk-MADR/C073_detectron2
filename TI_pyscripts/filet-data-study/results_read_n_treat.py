import pandas as pd
import seaborn as sns
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

df = pd.read_csv("/pers_files/Combined_final/Filet/output/results_pd-2021-5-12-19-50.csv")

sns.set(style="darkgrid")

x =  sns.lineplot(data=df, x="size", y="AP",hue="w20",ci=90)
x.set_xscale('log')
x.set_xticks(df['size'].unique())
x.set_xticklabels(df['size'].unique())

plt.show()