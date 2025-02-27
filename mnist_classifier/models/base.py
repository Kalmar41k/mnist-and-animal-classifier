from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    This is an abstract class that defines the interface for different MNIST classifiers.
    It enforces that all subclasses must implement the 'train' and 'predict' methods.
    
    Methods:
        train(X_train, y_train): Trains the model with the provided training data and labels.
        predict(X_test): Makes predictions on the test data using the trained model.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass