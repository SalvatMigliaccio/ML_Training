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
# print("Features after handling missing data:")
# print(x)

# Encoding categorical data
#encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])] , remainder='passthrough')
x = np.array(ct.fit_transform(x))
# print("Features after encoding categorical data:")
# print(x)

#encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# print("Target variable after encoding:")
# print(y)


#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print("Training set features (x_train):")
print(x_train)
print("Training set target variable (y_train):")
print(y_train)
print("Test set features (x_test):")
print(x_test)
print("Test set target variable (y_test):")
print(y_test)

#feature scaling with standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
print("-" * 50)
print ("Features after feature scaling:")
print("Training set features (x_train) after scaling:")
print(x_train)
print("Test set features (x_test) after scaling:")
print(x_test)