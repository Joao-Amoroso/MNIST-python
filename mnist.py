import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.metrics
import joblib
mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(len(x_train), 784)
x_test = x_test.reshape(len(x_test), 784)

mlp = MLPClassifier(random_state=120)

mlp.fit(x_train, y_train)
joblib.dump(mlp, "MLPclassifier.joblib")

"""
TODO
- transformar dataset: pondo em 0 e 255 e talvez normalizando(ver como funciona com normal e sem)
- passar o dataset para um ficheiro de texto para nao estar sempre a correr
- criar cnn com novo datasset
- criar interface com pygame
- fazer grid 28x28
- consigo desenhar nessa grid
- clicar em space e programa dar output
"""
