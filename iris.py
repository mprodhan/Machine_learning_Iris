from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris = load_iris()
# print(iris)
# The iris variable prints out all data on iris. They come from the scikit-learn library.

X = iris.data
# print(X)
# data variable only gets the data on the list array. X(data), also a numpy data array.
y = iris.target
# print(y)
# The target represents the most well known types of iris flower known to exist. They are categorized as 
# 0,1,2, in which each number is a type of iris that exists. y(target)

feature_names = iris.feature_names
target_names = iris.target_names
print(feature_names)
print(target_names)

# Traning data for the computer to learn.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
print(X_train.shape)
print(X_test.shape)
# Both of these print statement gives us, (rows, 4(dimensionality)).
# (90, 4)  means 90 rows from the test_size over its dimensionality.

knn = KNeighborsClassifier(n_neighbors=3)
train = knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))

sample = [
    [3,5,4,2],
    [2,3,5,4]
]

predictions = knn.predict(sample)
predict_species = [iris.target_names[p] for p in predictions]
print(" predictions : ", predict_species)