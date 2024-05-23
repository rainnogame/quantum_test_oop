from src.classifiers.base import DigitClassificationInterface
from src.classifiers.cnn import CNNDigitsClassifier
from src.classifiers.rand import RandomClassifier
from src.classifiers.rf import RandomForestDigitsClassifier


class ClassifiersFactory:
    """
    Factory for a classification algorithms
    """

    @staticmethod
    def from_name(
            algorithm_name: str, classifier_parameters: dict = None
    ) -> DigitClassificationInterface:
        """
        Create Digits classification algorithms based on classification algorithm name.
        :param algorithm_name: Name of the algorithm
        :param classifier_parameters: Parameter for algorithm initialization
        :return: Classification algorithm which follows DigitClassificationInterface
        """
        if classifier_parameters is None:
            classifier_parameters = {}

        match algorithm_name:
            case "rf":
                return RandomForestDigitsClassifier(**classifier_parameters)
            case "rand":
                return RandomClassifier(**classifier_parameters)
            case "cnn":
                return CNNDigitsClassifier(**classifier_parameters)
            case _:
                raise ValueError(f"Unknown algorithm name: {algorithm_name}")
