import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data = pd.read_csv("data.csv")

#Give a ranking of the sorting implementations by speed, including which ones could not be distinguished. (i.e. which pairs could our experiment not conclude had different running times?)
#adapted from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
p_value = stats.f_oneway(data.qs1, data.qs2, data.qs3, data.qs4, data.qs5, data.merge1, data.partition_sort)[1]
print(p_value) #pvalue=1.8811744756043763e-19 -> this means that there is at least one differnt pairs

#https://ggbaker.ca/data-science/slide-content/stats-tests.html#/posthoc-6
data_melt = pd.melt(data)
posthoc = pairwise_tukeyhsd(data_melt['value'], data_melt['variable'], alpha=0.5)
print(posthoc)

fig = posthoc.plot_simultaneous()
fig.show()
fig.savefig("fig")

print(data.mean(axis=0).sort_values())
