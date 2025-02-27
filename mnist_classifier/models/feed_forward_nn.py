from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from models.base import MnistClassifierInterface

class FeedForwardNNMnistClassifier(MnistClassifierInterface):
    """
    A Feedforward Neural Network (FNN) classifier for the MNIST dataset that inherits from MnistClassifierInterface.
    
    This class implements a simple Feedforward Neural Network with one hidden layer for classifying MNIST images.
    
    Methods:
        train(X_train, y_train, epochs=10, batch_size=32): Trains the FNN model on the provided training data.
        predict(X_test): Makes predictions using the trained FNN model on the test data.
    """
    def __init__(self, input_size, hidden_size=128, output_size=10):
        """
        Initializes the feedforward neural network model with the specified input size, hidden layer size, 
        and output size.
        
        The FNN consists of the following layers:
            - Dense: A fully connected layer with a hidden size (128 by default) and ReLU activation.
            - Dense: Output layer with softmax activation for multi-class classification.
        
        Parameters:
            input_size (int): The number of input features (flattened image size for MNIST is 784).
            hidden_size (int): The number of units in the hidden layer (default is 128).
            output_size (int): The number of output classes (default is 10 for MNIST).
        """
        self.model = Sequential([
            Dense(hidden_size, activation='relu', input_shape=(input_size,)),
            Dense(output_size, activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1)