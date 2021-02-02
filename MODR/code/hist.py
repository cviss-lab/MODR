import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('raw_pred.csv', index_col=0)
arr = df[df.columns[1:]].values.flatten()
plt.hist(arr, bins=100)
plt.yscale('log')
plt.show()
