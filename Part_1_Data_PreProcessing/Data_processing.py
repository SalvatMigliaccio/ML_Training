import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

#import the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values  # Target variable

# print("Features (x):")
# print(x)
# print("Target variable (y):")
# print(y)

# Handling missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])
print("Features after handling missing data:")
print(x)

# Encoding categorical data
#encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])] , remainder='passthrough')
x = np.array(ct.fit_transform(x))
print("Features after encoding categorical data:")
print(x)
