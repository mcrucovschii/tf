import numpy as np
import matplotlib.pyplot as plt
"""
greyhounds = 500
labs = 500

grey_height = 28+ 4 * np.random.randn(greyhounds)
lab_height = 28+ 4 * np.random.randn(labs)

plt.hist = ([grey_height, lab_height], stacked=True, color=['r','b'])
plt.show()

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
""""

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                        ])
model.summar ()
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit (train_images,train_labels, epochs=5)

