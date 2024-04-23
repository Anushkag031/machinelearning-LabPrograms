from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from reg import eval
import numpy as np
from sklearn import datasets

iris=datasets.load_iris()
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svr=SVR()
svr.fit(X_train,y_train)
y_pred=svr.predict(X_test)
eval(y_test,y_pred)
