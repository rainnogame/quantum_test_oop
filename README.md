## Goal

You have 3 different models to solve MNIST (handwritten digits database) classification
problem:

- Convolutional Neural Network (any architecture, any framework);
    - Input: tensor 28x28x1;
- Random Forest classifier;
    - Input: 1-d numpy array of length 784 (28x28 pixels);
- Model that provides random value (for simplicity) as a result of classification;
    - Input: 10x10 numpy array, the center crop of the image.

The goal is to build a DigitClassifier model that takes an algorithm as an input parameter. Possible values for the
algorithm are: cnn, rf, rand for the three models described above.

There is NO need to implement a training function inside DigitClassifier and focus on the quality of the model, just
raise a Not implemented exception. We need to focus only on the predict function that takes a 28x28x1 image as input and
provides a single integer value as output.

Ideally, the solution should contain:

- Interface for models like Convolutional Neural Network, Random Forest classifier, Random model. Potentially other
  developers will develop new models, so we need to have an interface for them. Let’s call it
  DigitClassificationInterface.
- 3 classes (1 for each model) that implement DigitClassificationInterface.
- DigitClassifier, which takes as an input parameter the name of the algorithm and provides predictions with exactly the
  same structure (inputs and outputs) not depending on the selected algorithm.

## Implementation notes

Code structure

- `digits_classifiers/classifiers/base.pt` - contains the interface for classification models
- `digits_classifiers/classifiers/cnn.py` - contains the implementation of the CNN model
- `digits_classifiers/classifiers/rf.py` - contains the implementation of the Random Forest model
- `digits_classifiers/classifiers/rand.py` - contains the implementation of the Random model
- `digits_classifiers/digit_classifier.py` - contains the implementation of the DigitClassifier
- `digits_classifiers/classifiers_factory.py` - contains the factory for the classifiers

Implemented under `python v3.12`

To install requirements:

```bash
pip install -r requirements.txt
```

To run tests:

```bash
pytest
```

To add new model you need to

- Extend `src.classifiers.base.DigitClassificationInterface`
- Add model as an option to a factory `from src.classifiers_factory.ClassifiersFactory`

## Important notes

- As a source dataset I selected `mnist_784` data from `sklearn.fetch_openml('mnist_784')`
- For a convenience models training and prediction are based on batch input
- Despite the fact that in task it's said to use `28x28x1` input, I used `1x28x28` seems it's more convenient as for me.
  I don't think this change in the rules is important
- Written test is artificial and cannot be considered as a production-ready test
- A lot if important staff for a production-ready code were skipped
    - Logging
    - Unit tests
    - Configurations
    - Input checks
    - etc