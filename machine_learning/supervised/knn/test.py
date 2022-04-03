import sklearn
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("data/car.data")
print(data.head())  # To check if our data is loaded correctly


le = preprocessing.LabelEncoder()

buying =le.fit_transform(list(data["buying"]))
maint =le.fit_transform(list(data["maint"]))
doors =le.fit_transform(list(data["buying"]))
persons =le.fit_transform(list(data["persons"]))
lug_boot =le.fit_transform(list(data["lug_boot"]))
safety =le.fit_transform(list(data["safety"]))
cls =le.fit_transform(list(data["class"]))
clsValues = le.classes_

features =list(zip(buying,maint,doors,persons,lug_boot,safety))
labels = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels, test_size=0.1)

neighbors = 9
model = KNeighborsClassifier(n_neighbors=neighbors)
model.fit(x_train, y_train)
accuracy=model.score(x_test,y_test)
print(accuracy)
predict = "class"

predicted = model.predict(x_test)

for x in range(len(predicted)):
    print("Predicted: ", clsValues[predicted[x]], "Data: ", x_test[x], "Actual: ", clsValues[y_test[x]])


