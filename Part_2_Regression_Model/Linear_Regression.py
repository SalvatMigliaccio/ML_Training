import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values  # Target variable

# splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print("Training set features (x_train):")
# print(x_train)
# print("Training set target variable (y_train):")
# print(y_train)
# print("Test set features (x_test):")
# print(x_test)
# print("Test set target variable (y_test):")
# print(y_test)

#training the Linear Regression model on the Training set
#the test set is not used in training, it is only for evaluation
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the Test set results
y_pred = regressor.predict(x_test)
print("Predicted values for the test set:")
print(y_pred)
print("Actual values for the test set:")
print(y_test)
# Visualizing both Training and Test set results side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Training set
axes[0].scatter(x_train, y_train, color='red')
axes[0].plot(x_train, regressor.predict(x_train), color='blue')
axes[0].set_title('Salary vs Experience (Training set)')
axes[0].set_xlabel('Years of Experience')
axes[0].set_ylabel('Salary')

# Test set
axes[1].scatter(x_test, y_test, color='red')
axes[1].plot(x_test, y_pred, color='blue')
axes[1].set_title('Salary vs Experience (Test set)')
axes[1].set_xlabel('Years of Experience')
axes[1].set_ylabel('Salary')

plt.tight_layout()
plt.show()
