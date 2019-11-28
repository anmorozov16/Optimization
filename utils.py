import numpy as np


class ModelSimulationTarget:
    def predict(self, x):
        return [np.random.random()]


class ModelOptimization:
    def predict(self, x):
        a = np.array([1, 0, 0])
        np.random.shuffle(a)
        return a


def load_model(path, compile):
    return ModelSimulationTarget()


def normalization(x):
    x = np.array(x)
    std = x.std()
    if std == 0:
        return np.full_like(x, 1.)
    return (x - x.mean()) / x.std()

