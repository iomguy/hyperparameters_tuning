import numpy as np
from parse_data import PData
from useful_functions import unison_shuffled_copies
import time
import os
import warnings
import logging
from neuro_evolution import evolution
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
warnings.filterwarnings('ignore')


def create_model(input_shape_X,
                 dense_hidden_layers_amount,
                 dense_neurons_on_layer_amounts,
                 dense_activation_types,
                 dropout_values,
                 metrics,
                 loss,
                 optimizer):
    """Function to construct a model, required for KerasClassifier"""

    # create model
    model = Sequential()

    # первый полносвязный слой
    model.add(Dense(dense_neurons_on_layer_amounts,
                    activation=dense_activation_types,
                    input_shape=input_shape_X))

    # скрытые слои
    for i in range(0, dense_hidden_layers_amount):
        model.add(
            Dense(dense_neurons_on_layer_amounts,
                  activation=dense_activation_types))
        model.add(Dropout(dropout_values))

    # обычный линейный нейрон в конце
    model.add(Dense(1))
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model


if __name__ == '__main__':
    # disables tensorflow debug info
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    start_time = time.time()
    prev_time = start_time
    curr_time = start_time

    # prepare data
    X, Y = PData("AllData.txt")
    # X, Y = unison_shuffled_copies(X, Y)

    # TrainSize = np.ceil(len(X) * 0.8).astype(int)
    #
    # X = np.array_split(X, [TrainSize], axis=0)
    # Y = np.array_split(Y, [TrainSize], axis=0)

    x_train = X[0]
    y_train = Y[0]
    x_test = X[1]
    y_test = Y[1]

    # create model
    # non-zero verbose parameter shows the progress for each net epoch
    #model = KerasRegressor(build_fn=create_model, epochs=50, verbose=0)


    # Задаём варьируемые гиперпараметры системы

    # input_shape для полносвязного слоя зависит от формы x_train, поэтмоу он тоже подаётся как вариант
    # параметра системы, просто один вариант
    input_shape_X_list = X.shape
    # варианты числа скрытых слоёв
    dense_hidden_layers_amount_list = [2, 3]
    # варианты числа нейронов для любого слоя (первого или скрытого)
    dense_neurons_on_layer_amount_list = [32, 64, 128, 256]
    # варианты функции активации для любого слоя (первого или скрытого)
    dense_activation_type_list = ["relu", "tanh", "linear", "sigmoid"]
    # варианты значений дропаута для каждого скрытого слоя
    dropout_list = [0, 0.1, 0.2, 0.3]
    # варианты значения batch для всей сети
    batch_size_list = [8, 16, 32]
    results_hyperparameters_file_name = "results_hyperparameters_neuro_evolution.txt"
    logging.basicConfig(filename="neuro_eval.log", level=logging.INFO)

    # TODO: переведи результаты в Pandas, сохраняй в .csv
    with open(results_hyperparameters_file_name, "w") as results_file:
        # тут общее количество слоёв, не скрытых
        params = {
            "epochs": [50],
            "batch_size": batch_size_list,
            "n_hidd_layers": [3],
            "n_neurons": [32, 64, 128, 256],
            "dropout": [0, 0.1, 0.2, 0.3],
            "optimizers": ["adam"],
            "activations": ["relu", "tanh", "linear", "sigmoid"],
            "last_layer_activations": ["sigmoid"],
            "losses": ["mse"],
            "metrics": ["mae"],
            "estimator": [create_model]
        }

        search = evolution.NeuroEvolution(generations=10, population=10, params=params)
        # наилучшая сеть ищется с точки зрения metrics
        search.evolve(X, Y)

        logging.info("Best metrics: {}, Best params: {}".
                     format(search.best_params.accuracy, search.best_params.network))
        logging.info("in {0} seconds".format(time.time()-start_time))