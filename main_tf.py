import numpy as np
import matplotlib.pyplot as plt
"""
greyhounds = 500
labs = 500

grey_height = 28+ 4 * np.random.randn(greyhounds)
lab_height = 28+ 4 * np.random.randn(labs)

plt.hist = ([grey_height, lab_height], stacked=True, color=['r','b'])
plt.show()
"""

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = .5)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit (X_train, y_train)

predictions = my_classifier.predict(X_test)
print (predictions)

from sklearn.metrics import accuracy_score
print ("Accuracy is ", accuracy_score(y_test,predictions))