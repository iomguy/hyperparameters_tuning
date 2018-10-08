import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_validate
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(patience=5)


def compile_model(input_dim):
    """

    :param network dict: dictionary with network parameters
    :param input_shape tuple: tuple with tradin data shape
    :return: compiled model
    """
    network = input_dim['network']
    input_shape = input_dim['input_shape']
    nb_hidd_layers = network.get('n_hidd_layers', 2)
    nb_neurons = network.get('n_neurons', 10)
    activation = network.get('activations', 'sigmoid')
    optimizer = network.get('optimizers', 'adam')
    metrics = [network.get('mertics', 'accuracy')]
    loss = network.get('losses', 'binary_crossentropy')

    dropout = network.get('dropout', 1)

    model_constructor = network.get('estimator', Sequential())

    model = model_constructor(input_shape_X=input_shape,
                              dense_hidden_layers_amount=nb_hidd_layers,
                              dense_neurons_on_layer_amounts=nb_neurons,
                              dense_activation_types=activation,
                              dropout_values=dropout,
                              loss=loss,
                              metrics=metrics,
                              optimizer=optimizer)
    # model = Sequential()
    #
    # model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
    # for i in range(nb_hidd_layers):
    #     model.add(Dense(nb_neurons, activation=activation))
    #     model.add(Dropout(network.get('dropout', 1)))
    #
    # model.add(Dense(
    #     network.get('last_layer_neurons', 1),
    #     activation=network.get('last_layer_activations', 'sigmoid'),
    # ))
    #
    # model.compile(loss=network.get('losses', 'binary_crossentropy'), optimizer=optimizer,
    #               metrics=[network.get('mertics', 'accuracy')])

    return model


def train_and_score(network, X, Y):
    # TODO: добавь сюда shuffle, вместо того, чтобы подавать уже заданные x_train, y_train, x_test, y_test
    """

        :param network dict: dictionary with network parameters
        :param x_train array: numpy array with features for traning
        :param y_train array: numpy array with labels for traning
        :param x_test array: numpy array with labels for test
        :param y_test array: numpy array with labels for test
        :return float: score
        """
    # skf = KFold(n_splits=2, shuffle=True)
    # # range(1) stops "for" after the 1-st iteration
    # for _, (train_index, test_index) in zip(range(1), skf.split(X, Y)):
    #     x_train, x_test = X[train_index], X[test_index]
    #     y_train, y_test = Y[train_index], Y[test_index]


    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    wrapped_model = KerasRegressor(build_fn=compile_model,
                                   input_dim={"network": network, "input_shape": (x_train.shape[1],)},
                                   epochs=50,
                                   verbose=network.get('verbose', 0),
                                   batch_size=network.get('batch_size', 128))

    # тут для кросс-валидации по дефолту используется KFold, но непонятно, делает ли он Shuffle
    cv_scoring = network.get('cv_scoring', 'neg_mean_absolute_error')
    cv_results = cross_validate(wrapped_model, X, Y, cv=5, n_jobs=-1, scoring=cv_scoring)
    result = np.mean(cv_results["test_score"])

    # # If no cross-validation, just shuffle
    # model = compile_model({"network": network, "input_shape": (x_train.shape[1],)})
    # model.fit(x_train, y_train,
    #           batch_size=network.get('batch_size', 128),
    #           epochs=10000,  # using early stopping, so no real limit
    #           verbose=network.get('verbose', 0),
    #           validation_data=(x_test, y_test),
    #           callbacks=[early_stopper])
    #
    # score = model.evaluate(x_test, y_test, verbose=0)
    # result = score[1]  # 1 is metrics. 0 is loss.

    return result
