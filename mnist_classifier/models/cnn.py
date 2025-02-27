from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
from models.base import MnistClassifierInterface

class CNNMnistClassifier(MnistClassifierInterface):
    """
    A CNN-based classifier for the MNIST dataset that inherits from the MnistClassifierInterface.
    
    This class implements a Convolutional Neural Network (CNN) to classify MNIST images.
    
    Methods:
        train(X_train, y_train, epochs=10, batch_size=32): Trains the CNN model on the provided training data.
        predict(X_test): Makes predictions using the trained CNN model on the test data.
    """
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        """
        Initializes the CNN model with the specified input shape and number of output classes.
        
        The CNN consists of the following layers:
            - Conv2D: A convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation.
            - MaxPooling2D: A max-pooling layer with a 2x2 pool size.
            - Flatten: Flattens the 2D output to a 1D vector.
            - Dense: Fully connected layer with 128 units and ReLU activation.
            - Dense: Output layer with a number of units equal to the number of classes (10 for MNIST), and softmax activation for multi-class classification.
        
        Parameters:
            input_shape (tuple): The shape of the input data (default is (28, 28, 1) for MNIST).
            num_classes (int): The number of classes in the dataset (default is 10 for MNIST).
        """
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1)