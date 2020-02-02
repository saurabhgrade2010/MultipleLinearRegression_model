import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data.csv')
dataset.isnull().sum()
dataset.info()

dataset.describe(include = 'all')
print(dataset.dtypes)

dataset = dataset.drop(['id','date'], axis = 1)

sns.pairplot(dataset)

X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values



#splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#plt.scatter(X_train,y_train,color= 'green')
plt.plot(X_train,regressor.predict(X_train),color='blue')
# Predicting the Test set results
y_pred = regressor.predict(X_test)

import statsmodels.api as sm

X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]
ols = sm.OLS(endog = y,exog = X_opt).fit()
ols.summary()


X_opt = X[:, [0, 1, 2, 3,5,6,7,8,9,10,11,12,13,14,15,16,17]]
ols = sm.OLS(endog = y,exog = X_opt).fit()
ols.summary()




def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((21613,19)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]
X_Modeled = backwardElimination(X_opt, SL)


