import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


#non si applica feature scaling, perchè non ci sono equazioni 
#training the decision tree on the whole dataset
regressor = DecisionTreeRegressor(random_state=0) #fixing the value
regressor.fit(x,y)

results = regressor.predict([[6.5]])
print(results)

X_grid = np.arange(min(x), max(x), 0.01) #to have a better resolution
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()