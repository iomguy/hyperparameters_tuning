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


# Function to create model, required for KerasClassifier
def create_model(input_shape_X, dense_type, dropout_value):
    # create model
    model = Sequential()
    model.add(Dense(32, activation=dense_type, input_shape=input_shape_X))
    # model.add(Flatten())
    model.add(Dropout(dropout_value))
    # param_dropout_grid = dict(dropout_rate=dropout_variants, batch_size=batch_size_variants)
    model.add(Dense(1))  # обычный линейный нейрон
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if __name__ == '__main__':
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

    input_shape_X_variants = [(x_train.shape[1],)]
    dense_layers_N = [3]
    dense_N_variants = [32, 64, 128]
    dense_type_variants = ["relu", "tanh", "softmax", "linear", "sigmoid"]
    dropout_variants = [0, 0.15, 0.25, 0.35]
    batch_size_variants = [16]

    param_grid = dict(batch_size=batch_size_variants,
                      dense_type=dense_type_variants,
                      input_shape_X=input_shape_X_variants,
                      dropout_value=dropout_variants)

    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=-1)
    grid_result = grid.fit(x_train, y_train)

    # summarize results
    print("Best: {0} using {1}".format(grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("mean: {0}, std:{1} with: {2}".format(mean, stdev, param))