import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X= np.array(dataset.iloc[:, 1:2])
print(X)
y= np.array(dataset.iloc[:, 2])
print(y)

'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)'''

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =6)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color = 'skyblue')
plt.plot(X, lin_reg.predict(X), color = 'red')
plt.title('Truth or bluff(LinearRegression)')
plt.xlabel('Position level')
plt.ylabel('Salary')


plt.scatter(X, y, color = 'skyblue')
plt.plot(X, lin_reg_2.predict( poly_reg.fit_transform(X)), color = 'red')
plt.title('Truth or bluff(LinearRegression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

lin_reg_2.predict( poly_reg.fit_transform(6.5))