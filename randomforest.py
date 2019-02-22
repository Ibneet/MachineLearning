import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X= np.array(dataset.iloc[:, 1:2])
print(X)
y= np.array(dataset.iloc[:, 2:3])
print(y)

'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)'''

from sklearn.ensemble import RandomForestRegressor 
regressor = RandomForestRegressor(n_estimators=10000, random_state=0)
regressor.fit(X, y) 

X_grid = np.arange(min(X),max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = 'black')
plt.plot(X_grid, regressor.predict(X_grid), color = 'orange')
plt.title('Truth or bluff(Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

y_pred = regressor.predict(6.5)