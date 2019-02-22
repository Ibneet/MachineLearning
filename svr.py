import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X= np.array(dataset.iloc[:, 1:2])
print(X)
y= np.array(dataset.iloc[:, 2:3])
print(y)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)


plt.scatter(X, y, color = 'skyblue')
plt.plot(X,regressor.predict(X), color = 'red')
plt.title('Truth or bluff(SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')

y_predict= sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))