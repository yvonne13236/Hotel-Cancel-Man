#https://www.kdnuggets.com/2019/10/easily-deploy-machine-learning-models-using-flask.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd


f = pd.read_csv('dk.csv')

#get x and y 
x=f[['uvIndex']]
y=f[['guests']]


#split into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Model initialization
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(X_train, y_train)



pickle.dump(regression_model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[4]]))