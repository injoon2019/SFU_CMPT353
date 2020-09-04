import numpy as np

data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']

#numpy.argmin(a, axis = None, out=None) returns the indicies of the minimum values along an axis
#axis = 0 means column and axis = 1 means row
lowest_precipitation = np.argmin(np.sum(totals, 1))
print('Row with lowest total precipitation:')
print(lowest_precipitation)

average_precipitation = np.divide(np.sum(totals, 0), np.sum(counts, 0)) 
print(average_precipitation)

average_precipitation_city = np.divide(np.sum(totals,1), np.sum(counts, 1))
print("Average precipitation in each city:")
print(average_precipitation_city)

#Review!
#each_quarter = np.reshape(totals, (len(totals) ,4,3))
each_quarter = np.reshape(totals, (len(totals) ,4,3)).sum(axis=2)
print("Quarterly precipitation totals:")
print(each_quarter)
