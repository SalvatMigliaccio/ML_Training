import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

#Data pre-processing 
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values  # Target variable
print(x)
print("-" * 50)


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = ct.fit_transform(x)
print(x)
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(X_train)
print("-" * 50)
print(X_test)
print("-" * 50)
print(y_train)
print("-" * 50)
print(y_test)
print("-" * 50)
#training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
print("Predicted values:")
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(y_pred)
print("-" * 50)
print("actual values:")
print(y_test)
print("-" * 50)

print("Concatenated results:")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))