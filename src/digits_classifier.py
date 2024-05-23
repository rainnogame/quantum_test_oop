from src.classifiers_factory import ClassifiersFactory
import numpy as np


class DigitClassifier:
    """
    Digits classifier to prediction digits from mnist data
    """

    def __init__(self, algorithm_name: str):
        """
        :param algorithm_name: Name of the algorithm that should be used for classification
        """
        self.algorithm = ClassifiersFactory.from_name(algorithm_name)

    def train(self, mnist_data: np.ndarray, mnist_target: np.ndarray):
        """
        Train classification model for digits prediction
        :param mnist_data: Input training data of shape (-1, 784)
        :param mnist_target: Target labels (string)
        """
        # Convert data to a format required by classification model
        converted_data = self.algorithm.convert_mnist_data_input(mnist_data)
        converted_targets = self.algorithm.convert_mnist_data_target(mnist_target)
        self.algorithm.train(converted_data, converted_targets)

    def predict(self, mnist_data_sample: np.ndarray) -> str:
        """
        Predict single digit using classification model
        :param mnist_data_sample: Digit representation np.ndarray of shape (784)
        :return: Predicted digit (string)
        """
        return self.algorithm.predict(self.algorithm.convert_mnist_data_input(mnist_data_sample[np.newaxis, :]))[0]
