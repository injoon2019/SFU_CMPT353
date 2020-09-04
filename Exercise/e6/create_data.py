import time
from implementations import all_implementations
import numpy as np
import pandas as pd

data = pd.DataFrame(columns=["qs1", "qs2", "qs3", "qs4", "qs5", "merge1", "partition_sort"], index = np.arange(100))

#should be more than 40
for i in range(100):
    random_array = np.random.randint(-700, 700, 1000)
    for sort in all_implementations:
        st = time.time()    #start test_implementations
        res = sort(random_array)
        en = time.time()    #end time
        data.iloc[i][sort.__name__] = en-st

print(data)
data.to_csv('data.csv', index=False)
