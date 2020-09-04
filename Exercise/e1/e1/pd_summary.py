import pandas as pd

totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])
print("City with lowest total precipitation:")
print(totals.sum(axis=1).idxmin())

print("Average precipitation in each month:")
print(totals.sum(axis=0).div(counts.sum(axis=0)))

print("Average precipitation in each city:")
print(totals.sum(axis=1).div(counts.sum(axis=1)))
