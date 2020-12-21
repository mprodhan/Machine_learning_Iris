from sklearn.datasets import load_iris
from sklearn.model import train_test_split

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