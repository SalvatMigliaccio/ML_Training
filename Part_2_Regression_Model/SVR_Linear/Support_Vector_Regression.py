import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print("-" * 50)
print(y)
print("-" * 50)

y = y.reshape(len(y), 1)  # Reshaping y to be a 2D array for StandardScaler
#feature scaling of the whole X without splitting in train set and test set
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)
print(X)
print("-" * 50)
print(y)
print("-" * 50)
