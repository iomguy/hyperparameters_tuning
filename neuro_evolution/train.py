from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

early_stopper = EarlyStopping(patience=5)


def compile_model(network, input_shape):
    """

    :param network dict: dictionary with network parameters
    :param input_shape tuple: tuple with tradin data shape
    :return: compiled model
    """

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


def train_and_score(network, x_train, y_train, x_test, y_test):
    # TODO: добавь сюда shuffle, вместо того, чтобы подавать уже заданные x_train, y_train, x_test, y_test
    """

        :param network dict: dictionary with network parameters
        :param x_train array: numpy array with features for traning
        :param y_train array: numpy array with labels for traning
        :param x_test array: numpy array with labels for test
        :param y_test array: numpy array with labels for test
        :return float: score
        """

    model = compile_model(network, (x_train.shape[1],))

    model.fit(x_train, y_train,
              batch_size=network.get('batch_size', 128),
              epochs=10000,  # using early stopping, so no real limit
              verbose=network.get('verbose', 0),
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])
    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is metrics. 0 is loss.
