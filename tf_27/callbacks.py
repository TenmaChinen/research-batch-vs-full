from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from tensorflow.keras.callbacks import Callback
from numpy import inf

class ModelCallback(Callback):
    def __init__(self):
        self.__best_weights = None

    def on_train_begin(self, logs=None):
        self.__min_test_loss = inf

    def on_epoch_end(self, epoch, logs=None):
        test_loss = logs['loss']

        if (test_loss < self.__min_test_loss):
            self.__min_test_loss = test_loss
            self.__best_weights = [l.get_weights() for l in self.model.layers]

    def on_train_end(self, logs=None):
        for layer, weights in enumerate(self.__best_weights):
            if (weights):
                self.model.layers[layer].set_weights(weights)
