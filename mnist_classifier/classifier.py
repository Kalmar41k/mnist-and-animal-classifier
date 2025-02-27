from models import RandomForestMnistClassifier, FeedForwardNNMnistClassifier, CNNMnistClassifier

class MnistClassifier:
    """
    A classifier class for MNIST dataset that can use different algorithms (Random Forest, Feed-Forward NN, CNN) 
    based on the provided algorithm parameter.
    
    Methods:
        train(X_train, y_train): Trains the selected model using the provided training data.
        predict(X_test): Makes predictions using the selected model on the test data.
    """
    def __init__(self, algorithm, input_shape=(28, 28), **kwargs):
        """
        Initializes the classifier based on the algorithm specified.
        
        Parameters:
            algorithm (str): The algorithm to use for classification. Can be 'rf' for Random Forest, 
                              'nn' for Feed-Forward Neural Network, or 'cnn' for Convolutional Neural Network.
            input_shape (tuple): The input shape of the images (default is (28, 28) for MNIST).
            **kwargs: Additional keyword arguments passed to the respective classifier.
        """
        # Select the classifier based on the algorithm type
        if algorithm == "rf":
            self.classifier = RandomForestMnistClassifier(**kwargs)
        elif algorithm == "nn":
            input_size = input_shape[0] * input_shape[1]  # 28x28 -> 784
            self.classifier = FeedForwardNNMnistClassifier(input_size=input_size, **kwargs)
        elif algorithm == "cnn":
            input_shape = (*input_shape, 1)  # (28, 28) -> (28, 28, 1)
            self.classifier = CNNMnistClassifier(input_shape=input_shape, **kwargs)
        else:
            raise ValueError("Unknown algorithm. Choose from 'rf', 'nn', or 'cnn'.")

    def train(self, X_train, y_train):
        """
        Trains the selected classifier on the provided training data.
        
        Parameters:
            X_train (array-like): The training data (features).
            y_train (array-like): The training labels.
        """
        self.classifier.train(X_train, y_train)

    def predict(self, X_test):
        """
        Makes predictions for the test data using the selected classifier.
        
        Parameters:
            X_test (array-like): The test data (features).
        
        Returns:
            array-like: The predicted labels for the test data.
        """
        return self.classifier.predict(X_test)