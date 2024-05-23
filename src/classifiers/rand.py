import numpy as np

from src.classifiers.base import DigitClassificationInterface


class RandomClassifier(DigitClassificationInterface):
    """
    Digits classifier based on random suggestion and cropped input
    See DigitClassificationInterface for methods description
    """

    def convert_mnist_data_input(self, mnist_data_sample_input: np.ndarray) -> np.ndarray:
        """
        Convert input data from mnist format to model's required format
        :param mnist_data_sample_input:  np.ndarray of shape (-1, 784)
        :return: Input data in required format for a model. np.ndarray of shape (-1, 10, 10)
        """
        return mnist_data_sample_input.reshape(-1, 28, 28)[:, 9:19, 9:19]

    def convert_mnist_data_target(self, mnist_data_sample_target: np.ndarray):
        """
        Convert target data from mnist format to model's required format
        :param mnist_data_sample_target: 1d np.ndarray
        :return: Target data in required format for a model
        """
        return mnist_data_sample_target

    def train(self, input_data: np.ndarray, target: np.ndarray):
        """
        Train classification model
        :param input_data: Input training data (input should be a result of `convert_mnist_data_target` function)
        :param target: Input target labels
        """
        raise NotImplementedError

    def predict(self, input_data: np.ndarray) -> list[str]:
        """
        Predict digits (return random prediction)
        :param input_data: Prediction data (input should be a result of `convert_mnist_data_target` function)
        :return: List of predicted digits (strings)
        """
        return np.random.randint(0, 10, len(input_data)).astype(str).tolist()
