import abc

import numpy as np


class DigitClassificationInterface(abc.ABC):
    """
    Base interface for Digits classification. Should be implemented in order to extend functionality to the new model
    """

    @abc.abstractmethod
    def convert_mnist_data_input(self, mnist_data_sample_input: np.ndarray):
        """
        Convert input data from mnist format to model's required format
        :param mnist_data_sample_input:  np.ndarray of shape (-1, 784)
        :return: Input data in required format for a model
        """
        ...

    @abc.abstractmethod
    def convert_mnist_data_target(self, mnist_data_sample_target: np.ndarray):
        """
        Convert target data from mnist format to model's required format
        :param mnist_data_sample_target: 1d np.ndarray
        :return: Target data in required format for a model
        """
        ...

    @abc.abstractmethod
    def train(self, input_data, target):
        """
        Train classification model
        :param input_data: Input training data (input should be a result of `convert_mnist_data_target` function)
        :param target: Input target labels
        """
        ...

    @abc.abstractmethod
    def predict(self, input_data) -> list[str]:
        """
        Predict digits
        :param input_data: Prediction data (input should be a result of `convert_mnist_data_target` function)
        :return: List of predicted digits (strings)
        """
        ...
