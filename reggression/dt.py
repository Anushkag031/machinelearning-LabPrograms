from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from reg import eval
import numpy as np
from sklearn import datasets

iris=datasets.load_iris()
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)

eval(y_test,y_pred)
