import numpy as np
import pandas as pd

#Generate synthetic linear data: y = 2x + 1 + noise
np.random.seed(42)               
n = 100
x = np.random.uniform(-10, 10, n)
noise = np.random.normal(0, 1, n)   
y = 2 * x + 1 + noise

df = pd.DataFrame({'x': x, 'y': y})
df.to_csv('linear_data.csv', index=False)
