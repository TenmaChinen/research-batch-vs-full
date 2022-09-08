from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from numpy.random import rand, seed as npy_seed
from numpy import array, arange, hstack
from pandas import DataFrame, concat
from sklearn.datasets import load_iris

def get_xor_dataset(split=0.3, seed=1, n_noise=0):
    '''
            out : x_train , y_train , x_test , y_test , x_sample , y_sample
    '''
    def xor(a, b): return 1 if (((a < 0.5) and (b >= 0.5))
                                or ((b < 0.5) and (a >= 0.5))) else 0
    x_range = arange(0, 1, 0.05)

    inputs = array([[x_0, x_1] for x_0 in x_range for x_1 in x_range])
    outputs = array([xor(x_0, x_1) for x_0, x_1 in inputs]).reshape(-1, 1)

    x_sample = array([[x_0, x_1] for x_0 in range(2) for x_1 in range(2)])
    y_sample = array([xor(x_0, x_1) for x_0, x_1 in x_sample]).reshape(-1, 1)

    if (n_noise):
        npy_seed(seed)
        inputs = hstack([inputs, rand(inputs.shape[0], n_noise).round(1)])
        x_sample = hstack([x_sample, rand(4, n_noise).round(1)])

    result = train_test_split(
        inputs, outputs, test_size=split, shuffle=True, random_state=seed)
    x_train, x_test, y_train, y_test = result

    return x_train, y_train, x_test, y_test, x_sample, y_sample


def get_iris_dataset(split=0.3, seed=1):
    '''
        output : x_train , y_train , x_test , y_test
    '''
    iris_data = load_iris()

    df_inputs = DataFrame(iris_data.data, columns=iris_data.feature_names)

    iris_target = to_categorical(iris_data.target.reshape(-1, 1))
    df_outputs = DataFrame(iris_target, columns=iris_data.target_names)

    df_iris = concat(dict(x=df_inputs, y=df_outputs), axis=1)
    df_iris.x = df_iris.x / df_iris.x.max(axis=0)

    result = train_test_split(df_iris.x.values, df_iris.y.values, test_size=split, shuffle=True, random_state=seed)
    x_train, x_test, y_train, y_test = result

    return x_train, y_train, x_test, y_test