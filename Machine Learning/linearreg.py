import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("C:/Users/cpilat/Desktop/Projects/Python/tensorEnv/student-mat.csv", sep=';')

data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]

predict = "G3"



X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
best = 0

"""
for _ in range(30):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(X_train, y_train)
    acc = linear.score(X_test, y_test)
    print(acc)

    if acc > best: 
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""


pickle_in = open("studentmodel.pickle", "rb")   
linear = pickle.load(pickle_in)

print("Coefficient \n", linear.coef_)
print("Intercept \n", linear.intercept_)

predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])


style.use("ggplot")

p = "absences"


plt.scatter(data[p], data["G3"], alpha = 0.5)
plt.xlabel("P")
plt.ylabel("Final Grade")
plt.show()

