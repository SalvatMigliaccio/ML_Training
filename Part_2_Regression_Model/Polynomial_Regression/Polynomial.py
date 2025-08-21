import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values  # Features
y = dataset.iloc[:, -1].values  # Target variable
print(x)
print("-" * 50)
print(y)
print("-" * 50)

#training the linear Regression model on the whole dataset
lin_reg = LinearRegression()
lin_reg.fit(x, y)

#training the Polynomial Regression model on the whole dataset
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.subplots(1, 2, figsize=(14, 6))

#visualizing the Linear Regression results
plt.subplot(1, 2, 1)
plt.scatter(x,y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

#visualizing the Polynomial Regression results
plt.subplot(1, 2, 2)
plt.scatter(x,y, color='red')
plt.plot(x, lin_reg_2.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.tight_layout()
plt.show()

#visualizing the Polynomial Regression results with a higher resolution and smoother curve
x_grid = np.arange(min(x), max(x), 0.01)
plt.plot()
plt.scatter(x,y, color='red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid.reshape(-1, 1))), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.tight_layout()
plt.show()

#predicting a new result with linear Regression
level = 6.5
Response = lin_reg.predict([[level]])
print(f"Predicted salary for level {level}: {Response[0]}")
print("-" * 50)

#predicting a new result with Polynomial Regression
response_poly = lin_reg_2.predict(poly_reg.fit_transform([[level]]))
print(f"Predicted salary for level {level} (Polynomial Regression): {response_poly[0]}")
print("-" * 50)