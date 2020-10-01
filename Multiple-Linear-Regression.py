# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\nstej\ML prac\Multiple-Linear-Regression-master\50_Startups.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

#Convert the column into categorical columns

states=pd.get_dummies(X['State'],drop_first=True)

# Drop the state coulmn
X=X.drop('State',axis=1)

# concat the dummy variables
X=pd.concat([X,states],axis=1)

#splitting the dataset into the traning and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Fitting multiple linear regression to the traning set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)