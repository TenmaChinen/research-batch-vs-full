from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from nn.tf_27.utils import set_train_seed
from os import environ

environ['CUDA_VISIBLE_DEVICES'] = '-1'

def make_mlp_model(
        n_inputs, n_outputs, n_hidden=10, act_hidden='sigmoid', act_output='linear',
        loss='mean_squared_error', lr=0.01, decay=0.0, drop=None,
        metrics=None, model_path=None, seed=1
):
    ''' 
        act_hidden : { relu , sigmoid , softsign }
        act_output : { linear , sigmoid , softmax }
        loss : { mean_squared_error , categorical_crossentropy }
        metrics : [ 'mse' , ... ]
    '''

    set_train_seed(seed)

    random_normal = RandomNormal(mean=0, stddev=0.05, seed=seed)
    kernel_args = dict( use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal)

    hidden_1 = Dense(n_hidden, input_dim=n_inputs, activation=act_hidden, **kernel_args)
    output_1 = Dense(n_outputs, activation=act_output, **kernel_args)

    l_model = [hidden_1, output_1]

    if (drop):
        l_model.insert(1, Dropout(rate=drop, seed=seed))

    model = Sequential(l_model)
    model.compile(loss=loss, optimizer=Adam( learning_rate=lr, decay=decay), metrics=metrics)

    if (model_path):
        json_model = model.to_json()
        file = open(f'{model_path}.json', 'w')
        file.write(json_model)
        file.close()

    return model


def load_model(model_path):
    file = open(f'{model_path}.json', 'r')
    json_model = file.read()
    file.close()

    best_model = model_from_json(json_model)
    best_model.load_weights(f'{model_path}.h5')
