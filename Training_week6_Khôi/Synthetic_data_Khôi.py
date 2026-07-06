import numpy as np
import pandas as pd

def choose_function(x, kind):
    if kind =="linear":
        return 2*x + 1
    elif kind == "quadratic":
        return x**2 + 2*x + 3
    elif kind == "cubic":
        return x**3 + 3*x**2 - x + 2
    else:
        raise ValueError(kind)

np.random.seed(42)               
n = 300
x = np.random.uniform(-10, 10, n)

option = int(input("Choose function type: "))
match option:
    case 1: kind = "linear"
    case 2: kind = "quadratic"
    case 3: kind = "cubic"

y_clean = choose_function(x, kind)
noise_std = 0.05 * np.std(y_clean)   # noise = 5% of that function's own spread
noise = np.random.normal(0, noise_std, n)
y = y_clean + noise

df = pd.DataFrame({'x': x, 'y': y})
df.to_csv(f'{kind}_data.csv', index=False)

