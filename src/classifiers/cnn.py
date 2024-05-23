import torch
import numpy as np

from src.classifiers.base import DigitClassificationInterface


class Flatten(torch.nn.Module):
    """
    Torch module to flatten input data.
    """

    def forward(self, x: torch.Tensor):
        """
        Convert input tensor in flattened format
        :param x: Input tensor
        :return: Flattened tensor
        """
        return x.view(x.size(0), -1)


class CNNDigitsClassifier(DigitClassificationInterface):
    """
    Digits classifier based on CNN
    See DigitClassificationInterface for methods description
    """

    def __init__(self):
        """
        Created arbitrary CNN architecture
        """
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            Flatten(),
            torch.nn.Linear(14 * 14, 10),
            torch.nn.Softmax(dim=1)
        )

    def convert_mnist_data_target(self, mnist_data_sample_target: np.ndarray) -> torch.Tensor:
        """
        Convert target data from mnist format to model's required format
        :param mnist_data_sample_target: 1d np.ndarray
        :return: Target data in required format for a model. One-hot encoded tensor
        """
        return torch.nn.functional.one_hot(torch.Tensor(mnist_data_sample_target.astype(int)).to(torch.int64), 10)

    def convert_mnist_data_input(self, mnist_data_sample_input: np.ndarray) -> torch.Tensor:
        """
        Convert input data from mnist format to model's required format
        :param mnist_data_sample_input:  np.ndarray of shape (-1, 784)
        :return: Input data in required format for a model. Tensor of shape (-1, 1, 28, 28)
         """
        return torch.tensor(mnist_data_sample_input, dtype=torch.float).reshape(-1, 1, 28, 28)

    def train(self, input_data: torch.Tensor, target: torch.Tensor):
        """
        Train classification model
        :param input_data: Input training data (input should be a result of `convert_mnist_data_target` function)
        :param target: Input target labels
        """
        raise NotImplementedError

    def predict(self, input_data: torch.Tensor) -> list[str]:
        """
        Predict digits
        :param input_data: Prediction data (input should be a result of `convert_mnist_data_target` function)
        :return: List of predicted digits (strings)
        """
        with torch.no_grad():
            return self.model(input_data).argmax(dim=1).numpy().astype(str).tolist()
