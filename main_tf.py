import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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

user_id = tf.feture_column.embedding_coumn(user_id,10)

columns = [uid_embedding,
           tf.feature_column.numeric_column('visits'),
           tf.feature_column.numeric_column('clicks')
]

feature_layer = tf.keras.layers.DenseFeatures(columns)
#model = tf.keras.models.Sequential

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                        ])
model.summary ()
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit (train_images,train_labels, epochs=5)

model.save ('/var/model', save_format='tf')

new_model = tf.keras.model.load_model('/var/model')
new_model.summary()

weights = tf.Variable([tf.random.normal()])

while True:
    with tf.GradientTape() as g:
        loss = g.compute_loss(weights)
        gradient = g.gradient (loss,weights)
    weights = weights - g.lr * gradient

