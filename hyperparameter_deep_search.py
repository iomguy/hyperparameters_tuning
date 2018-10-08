from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.constraints import maxnorm

import numpy as np
from parse_data import PData
from useful_functions import unison_shuffled_copies
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import itertools
import time
import CVProgressBar


def create_model(input_shape_X,
                 dense_hidden_layers_amount,
                 dense_neurons_on_layer_amounts,
                 dense_activation_types,
                 dropout_values):
    """Function to construct a model, required for KerasClassifier"""

    # create model
    model = Sequential()

    # первый полносвязный слой
    model.add(Dense(dense_neurons_on_layer_amounts[0],
                    activation=dense_activation_types[0],
                    input_shape=input_shape_X))

    # скрытые слои
    for i in range(0, dense_hidden_layers_amount):
        model.add(
            Dense(dense_neurons_on_layer_amounts[i+1],
                  activation=dense_activation_types[i+1],
                  input_shape=input_shape_X))
        model.add(Dropout(dropout_values[i]))

    # обычный линейный нейрон в конце
    model.add(Dense(1))
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])

    return model


if __name__ == '__main__':
    # disables tensorflow debug info
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    start_time = time.time()
    prev_time = start_time
    curr_time = start_time

    # prepare data
    X, Y = PData("AllData.txt")
    X, Y = unison_shuffled_copies(X, Y)

    TrainSize = np.ceil(len(X) * 0.8).astype(int)

    X = np.array_split(X, [TrainSize], axis=0)
    Y = np.array_split(Y, [TrainSize], axis=0)

    x_train = X[0]
    y_train = Y[0]
    x_test = X[1]
    y_test = Y[1]

    # create model
    # non-zero verbose parameter shows the progress for each net epoch
    model = KerasRegressor(build_fn=create_model, epochs=50, verbose=0)


    # Задаём варьируемые гиперпараметры системы

    # input_shape для полносвязного слоя зависит от формы x_train, поэтмоу он тоже подаётся как вариант
    # параметра системы, просто один вариант
    input_shape_X_list = (x_train.shape[1],)
    # варианты числа скрытых слоёв
    dense_hidden_layers_amount_list = [3]
    # варианты числа нейронов для любого слоя (первого или скрытого)
    dense_neurons_on_layer_amount_list = [32, 64, 128, 256]
    # варианты функции активации для любого слоя (первого или скрытого)
    dense_activation_type_list = ["relu", "tanh", "linear", "sigmoid"]
    # варианты значений дропаута для каждого скрытого слоя
    dropout_list = [0, 0.1, 0.2, 0.3]
    # варианты значения batch для всей сети
    batch_size_list = [8, 16, 32]
    results_hyperparameters_file_name = "results_hyperparameters.txt"

    # TODO: переведи результаты в Pandas, сохраняй в .csv
    with open(results_hyperparameters_file_name, "w") as results_file:
        # тут вручную параметры перебираются для каждого числа слоёв
        for dense_hidden_layers_amount in dense_hidden_layers_amount_list:
            # перебираются все возможные сочетания вариантов параметров
            # по выбранному dense_hidden_layers_amount числу слоёв
            # TODO: можно сделать списки ниже одним генератором?
            # число нейронов на слое и функция активации задаются как для скрытых, так и для первого слоя,
            # поэтому их их число - dense_hidden_layers_amount + 1
            dense_neurons_on_layer_amount_list_for_curr_layers_amount = \
                tuple(itertools.combinations_with_replacement
                      (dense_neurons_on_layer_amount_list, dense_hidden_layers_amount + 1))
            dense_activation_type_list_for_curr_layers_amount = \
                tuple(itertools.combinations_with_replacement
                      (dense_activation_type_list, dense_hidden_layers_amount + 1))
            # дропауты задаются только после скрытых слоёв, поэтмоу их число - dense_hidden_layers_amount
            dropout_list_for_curr_layers_amount = \
                tuple(itertools.combinations_with_replacement
                      (dropout_list, dense_hidden_layers_amount))

            # задаём параметры сети, которые будем варьировать
            param_grid = dict(batch_size=batch_size_list,
                              input_shape_X=[input_shape_X_list],
                              dense_hidden_layers_amount=[dense_hidden_layers_amount],
                              dense_neurons_on_layer_amounts=dense_neurons_on_layer_amount_list_for_curr_layers_amount,
                              dense_activation_types=dense_activation_type_list_for_curr_layers_amount,
                              dropout_values=dropout_list_for_curr_layers_amount)

            # создаём объект, который будет варьировать параметры сети
            # TODO: попробуй ещё и RandomizedSearchCV с неким заданным числом итераций, вместо последовательного
            # grid = GridSearchCV(estimator=model,
            #                     param_grid=param_grid,
            #                     n_jobs=-1)

            # divides train set into "cv" part for the cross-validation (by default cv = 3)
            cv = 5
            # стырил обёртку, добавляющую прогрессбар для GridSearchCV на каждом слое
            # непонятно, какое число потоков n_jobs следует ставить. Значение -1 позволяет выбирать его автоматически
            # grid = CVProgressBar.GridSearchCVProgressBar(estimator=model,
            #                                              param_grid=param_grid,
            #                                              n_jobs=-1,
            #                                              cv=cv)

            grid = CVProgressBar.RandomizedSearchCVProgressBar(estimator=model,
                                                               param_distributions=param_grid,
                                                               n_jobs=-1,
                                                               cv=cv,
                                                               n_iter=250)

            # запускаем варьирование параметров для заданного числа слоёв, загоняем результаты в объекте
            grid_result = grid.fit(X, Y)

            # summarize results
            print("------------------------------------------------------------------------------------")
            print("For {} layers".format(dense_hidden_layers_amount))
            print("Best: {0} using {1}".format(grid_result.best_score_, grid_result.best_params_))

            curr_time = time.time()
            results_file.write("------------------------------------------------------------------------------------\n")
            results_file.write("For {} layers\n".format(dense_hidden_layers_amount))
            results_file.write("Best: {0} using {1}\n".format(grid_result.best_score_, grid_result.best_params_))
            results_file.write("in {0} seconds\n".format(curr_time - prev_time))
            prev_time = curr_time

            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']

            for mean, stdev, param in zip(means, stds, params):
                results_file.write("mean: {0}, std:{1} with: {2}\n".format(mean, stdev, param))
            print("Results for {0} layers are written to the file {1}"
                  .format(dense_hidden_layers_amount, results_hyperparameters_file_name))

        results_file.write(
            "-------------------------------------------------------------------------------------------------------\n")
        results_file.write("Fin in {} seconds!\n".format(curr_time - start_time))