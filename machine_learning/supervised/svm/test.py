import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(cancer.data, cancer.target, test_size=0.2)

classes = cancer.target_names

clf = svm.SVC(kernel="linear")
clf.fit(x_train,y_train)
y_prod = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_prod)
print(accuracy)