import tensorflow as tf
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from joblib import dump


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# TODO: alterar os parametros para obter melhores valores
mlp = MLPClassifier(random_state=120)
x_train_len = len(x_train)
x_test_len = len(x_test)

# without normalize acc 0.9621
x_train1 = x_train.reshape(x_train_len, 784)
x_test1 = x_test.reshape(x_test_len, 784)
"""

mlp.fit(x_train1, y_train)

dump(mlp,"MLPclassifierWithoutNormalization")
# with normalize acc 0.9738

x_train2 = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train_len, 784)
x_test2 = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test_len, 784)


mlp.fit(x_train2, y_train)
"""

# with normalize /255 acc 0.9777

x_train3 = x_train1/255.0
x_test3 = x_test1/255.0


mlp.fit(x_train3, y_train)
dump(mlp, "MLPclassifierWithNormalization255.joblib")
print(accuracy_score(y_test, mlp.predict(x_test3)))
