from builtins import type
import pytest
import numpy as np
from sklearn.datasets import fetch_openml
import torch
from src.digits_classifier import DigitClassifier


def test_classifier():
    # Load data
    mnist = fetch_openml('mnist_784')
    data = mnist['data'].values
    target = np.array(mnist['target'].values)

    # Create classifier "rand"
    digit_classifier = DigitClassifier('rand')

    # Check shape
    samples = digit_classifier.algorithm.convert_mnist_data_input(data[:10])
    assert samples[0].shape == (10, 10)
    assert type(samples) is np.ndarray

    # Check training and prediction
    with pytest.raises(NotImplementedError):
        digit_classifier.train(data[:10], target[:10])

    digit_classifier.predict(data[0])

    # Create classifier "rf"
    digit_classifier = DigitClassifier('rf')

    # Check shape
    samples = digit_classifier.algorithm.convert_mnist_data_input(data[:10])
    assert samples[0].shape == (784,)
    assert type(samples) is np.ndarray

    # Check training and prediction
    digit_classifier.train(data[:10], target[:10])
    digit_classifier.predict(data[0])

    # Create classifier
    digit_classifier = DigitClassifier('cnn')

    # Check shape
    samples = digit_classifier.algorithm.convert_mnist_data_input(data[:10])
    assert samples[0].shape == torch.Size([1, 28, 28])
    assert type(samples) is torch.Tensor

    # Check training and prediction
    with pytest.raises(NotImplementedError):
        digit_classifier.train(data[:10], target[:10])

    digit_classifier.predict(data[0])
