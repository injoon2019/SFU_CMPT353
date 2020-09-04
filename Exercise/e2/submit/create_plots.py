import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename1 = sys.argv[1]
filename2 = sys.argv[2]
file1 = pd.read_csv(filename1, sep=' ', header = None, index_col=1, names=['lang', 'page', 'views', 'bytes'])
file2 = pd.read_csv(filename2, sep=' ', header = None, index_col=1, names=['lang', 'page', 'views', 'bytes'])


#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
file1_view = file1.sort_values('views', ascending=False)['views']
file2_view = file2.sort_values('views', ascending=False)['views']
#Both dataframe have 'view', so we need to distinguish them.
file1_view.name = "view1"
file2_view.name = "view2"

#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
f1_f2 = pd.concat([file1_view, file2_view], axis=1).reset_index()

#print(f1_f2)
#print(file1_view)
#print(file2_view)
plt.figure(figsize=(15, 5)) #change the size to something sensible
plt.subplot(1,2, 1) #subplots in 1 row, 2 columns, select the first
plt.plot(file1_view.values)
plt.title('Popularity Distribution')
plt.xlabel("Rank")
plt.ylabel("Views")

plt.subplot(1,2,2)
plt.scatter(f1_f2['view1'], f1_f2['view2'], color='b')
#https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.xscale
plt.xscale('log')
plt.yscale('log')
plt.title('Daily Correlation')
plt.xlabel("Day 1 views")
plt.ylabel("Day 2 views")

plt.savefig('wikipedia.png')
