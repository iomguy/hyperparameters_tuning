from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.constraints import maxnorm

import numpy as np
from parse_data import PData
from myAcc import accur
from useful_functions import unison_shuffled_copies
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import os
import itertools


# Function to create model, required for KerasClassifier
def create_model(input_shape_X, dense_layers_amount, dense_neurons_on_layer_amounts, dense_activation_types, dropout_values):
    # create model
    model = Sequential()
    model.add(Dense(dense_neurons_on_layer_amounts[0], activation=dense_activation_types[0], input_shape=input_shape_X))
    model.add(Dropout(dropout_values[0]))
    # TODO: проверь, так ли ставятся дропауты
    for i in range(1, dense_layers_amount):
        model.add(
            Dense(dense_neurons_on_layer_amounts[i], activation=dense_activation_types[i], input_shape=input_shape_X))
        model.add(Dropout(dropout_values[i]))

    model.add(Dense(1))  # обычный линейный нейрон
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if __name__ == '__main__':
    # disables tensorflow debug info
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # prepare data
    X, Y = PData("AllData.txt")
    X, Y = unison_shuffled_copies(X, Y)

    TrainSize = np.ceil(len(X) * 0.8).astype(int)

    X = np.array_split(X, [TrainSize], axis=0)
    Y = np.array_split(Y, [TrainSize], axis=0)

    x_train =X[0]
    x_test = X[1]
    y_train =Y[0]
    y_test = Y[1]

    # create model
    model = KerasRegressor(build_fn=create_model, epochs=50, verbose=0)
    # define the grid search parameters

    input_shape_X_list = (x_train.shape[1],)
    # TODO: то, что слоёв один, значит, что мы ставим один слой после первого или что мы ставим в принципе один dense слой?
    dense_layers_amount_list = [2, 3]
    dense_neurons_on_layer_amount_list = [32, 64]
    dense_activation_type_list = ["relu", "tanh", "softmax", "linear", "sigmoid"]
    dropout_list = [0, 0.15]
    batch_size_list = [16]
    results_hyperparameters_file_name = "results_hyperparameters.txt"
    # тут придётся вручную последовательно перебирать для каждого числа слоёв
    # все возможные комбинации параметров для каждого слоя
    for dense_layers_amount in dense_layers_amount_list:
        # TODO: можно сделать списки ниже одним генератором?

        # TODO: НЕПРАВИЛЬНО! создавать список всех возможных комбинаций элементов из dense_neurons_on_layer_amount_list по dense_layers_amount штук!
        # dense_neurons_on_layer_amount_list_for_curr_layers_amount = [dense_neurons_on_layer_amount_list
        #                                                              for i in range(0, dense_layers_amount)]
        # dense_activation_type_list_for_curr_layers_amount = [dense_activation_type_list
        #                                                      for i in range(0, dense_layers_amount)]
        # dropout_list_for_curr_layers_amount = [dropout_list
        #                                        for i in range(0, dense_layers_amount)]

        dense_neurons_on_layer_amount_list_for_curr_layers_amount = \
            list(itertools.combinations_with_replacement(dense_neurons_on_layer_amount_list, dense_layers_amount))
        dense_activation_type_list_for_curr_layers_amount = \
            list(itertools.combinations_with_replacement(dense_activation_type_list, dense_layers_amount))
        dropout_list_for_curr_layers_amount = \
            list(itertools.combinations_with_replacement(dropout_list, dense_layers_amount))

        # TODO: проверь, столько ли нужно дропаутов

        param_grid = dict(batch_size=batch_size_list,
                          input_shape_X=[input_shape_X_list],
                          dense_layers_amount=[dense_layers_amount],
                          dense_neurons_on_layer_amounts=dense_neurons_on_layer_amount_list_for_curr_layers_amount,
                          dense_activation_types=dense_activation_type_list_for_curr_layers_amount,
                          dropout_values=dropout_list_for_curr_layers_amount)

        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            n_jobs=-1) #TODO: попробуй ещё и RandomizedSearchCV с неким заданным числом итераций
        grid_result = grid.fit(x_train, y_train)

        # summarize results
        print("-------------------------------------------------------------------------------------------------------")
        print("For {} layers".format(dense_layers_amount))
        print("Best: {0} using {1}".format(grid_result.best_score_, grid_result.best_params_))

        with open(results_hyperparameters_file_name, "w") as results_file:
            results_file.write(
                "-------------------------------------------------------------------------------------------------------\n")
            results_file.write("For {} layers\n".format(dense_layers_amount))
            results_file.write("Best: {0} using {1}\n".format(grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                results_file.write("mean: {0}, std:{1} with: {2}\n".format(mean, stdev, param))
            print("Results for {0} layers are written to the file {1}"
                  .format(dense_layers_amount, results_hyperparameters_file_name))
