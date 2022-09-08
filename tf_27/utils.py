from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from tensorflow import __version__ as tf_version
from numpy.random import seed as set_numpy_seed
from random import seed as set_native_Seed
from os import environ

print(f"Tensorflow Version : {tf_version}")  # 1.12.0

# environ['TF_CPP_MIN_LOG_LEVEL'] = '2' ;

if (tf_version == '2.7.0'):
    from tensorflow.keras.utils import set_random_seed as set_tf_seed


def set_train_seed(seed=3):
    environ['PYTHONHASHSEED'] = '0'
    environ['TF_DETERMINISTIC_OPS'] = '1'
    environ['TF_CUDNN_DETERMINISTIC'] = '1'
    set_numpy_seed(seed)
    set_native_Seed(seed)
    set_tf_seed(seed)


def get_hist_info(history, metric='loss'):
    '''
        inputs : { history , metric }
        output : min_loss, min_val_loss , l_loss , l_val_loss
        output : min_metric, min_val_metric , l_metric , l_val_metric
    '''

    val_metric = f'val_{metric}'

    l_metric = history[metric]
    l_val_metric = history[val_metric]

    min_val_metric = min(l_val_metric)
    min_val_metric_idx = l_val_metric.index(min_val_metric)
    min_metric = l_metric[min_val_metric_idx]

    return min_metric, min_val_metric, l_metric, l_val_metric
