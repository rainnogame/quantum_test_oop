from sklearn.ensemble import RandomForestClassifier
from src.classifiers.base import DigitClassificationInterface
import numpy as np


class RandomForestDigitsClassifier(DigitClassificationInterface):
    """
    Digits classifier based on Random Forest classifier and flattened input
    See DigitClassificationInterface for methods description
    """

    def __init__(self, *args, **kwargs):
        self.model = RandomForestClassifier(*args, **kwargs)

    def convert_mnist_data_target(self, mnist_data_sample_target: np.ndarray):
        """
        Convert target data from mnist format to model's required format
        :param mnist_data_sample_target: 1d np.ndarray
        :return: Target data in required format for a model
        """
        return mnist_data_sample_target

    def convert_mnist_data_input(self, mnist_data_sample_input: np.ndarray):
        """
        Convert input data from mnist format to model's required format
        :param mnist_data_sample_input:  np.ndarray of shape (-1, 784)
        :return: Input data in required format for a model. Return mnist samples as is
        """
        return mnist_data_sample_input

    def train(self, input_data: np.ndarray, target: np.ndarray):
        """
        Train classification model
        :param input_data: Input training data (input should be a result of `convert_mnist_data_target` function)
        :param target: Input target labels
        """
        self.model.fit(input_data, target)

    def predict(self, input_data: np.ndarray) -> list[str]:
        """
        Predict digits
        :param input_data: Prediction data (input should be a result of `convert_mnist_data_target` function)
        :return: List of predicted digits (strings)
        """
        return self.model.predict(input_data)
