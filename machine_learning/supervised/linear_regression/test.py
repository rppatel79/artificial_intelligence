import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model


data = pd.read_csv("data/student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

features = np.array(data.drop([predict], 1))
labels = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels,
                                                              test_size = 0.1)

# best = 0
# for _ in range(30):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, label,
#                                                                   test_size = 0.1)
#
#     linear_regression  = linear_model.LinearRegression()
#
#     linear_regression.fit(x_train, y_train)
#     accuracy = linear_regression.score(x_test, y_test)
#     print(accuracy)
#
#     if ( best < accuracy ):
#         with open("studentmodel-pickle","wb") as f:
#             pickle.dump(linear_regression, f)

with open("studentmodel-pickle","rb") as f:
    linear_regression = pickle.load(f)

print("Coo", linear_regression.coef_)
print("Intercept", linear_regression.intercept_)

predictions = linear_regression.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# Drawing and plotting model
plot = "studytime"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()