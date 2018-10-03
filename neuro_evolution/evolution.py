import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from tqdm import tqdm
import logging

from optimizer import Optimizer


class NeuroEvolution:
    def __init__(self, generations, population, params):
        """
        :param generations int: number of generation
        :param population int: number of population inside each generation
        :param params dict: dictionary with parameters
        """
        self._generations = generations
        self._population = population
        self._params = params
        self.networks = None
        self.best_params = None

    def evolve(self, X, Y):
        """
        Takes data for traning and data for test and iterate thought generations to find parameters with lowest error
        :param x_train array: array with features for traning
        :param y_train array: array with real values for traning
        :param x_test array: array with features for test
        :param y_test array: array with real values for test
        :return: None

        """
        optimizer = Optimizer(self._params)
        self._networks = list(optimizer.create_population(self._population))

        for generation in range(self._generations - 1):
            self._train_networks(X, Y, generation)
            self._networks = optimizer.evolve(self._networks)

        self._networks = sorted(self._networks, key=lambda x: x.accuracy, reverse=True)
        self.best_params = self._networks[0]

    def _train_networks(self, X, Y, generation):
        """
        Method for networks training
        :param x_train array: array with features for traning
        :param y_train array: array with real values for traning
        :param x_test array: array with features for test
        :param y_test array: array with real values for test
        :return: None
        """
        pbar = tqdm(total=len(self._networks))
        pbar.set_description("Generation: {}".format(generation))
        for network in self._networks:
            network.train(X, Y)
            pbar.update(1)
        pbar.close()

    def _get_average_accuracy(self, networks):
        """
        :param networks list: list of dictionaries
        :return float: mean accuracy per population
        """
        return sum([network.accuracy for network in networks]) / len(networks)
