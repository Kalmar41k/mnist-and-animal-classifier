from sklearn.ensemble import RandomForestClassifier
from models.base import MnistClassifierInterface

class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    A Random Forest classifier for the MNIST dataset that inherits from MnistClassifierInterface.
    
    This class implements a Random Forest model for classifying MNIST images.
    
    Methods:
        train(X_train, y_train): Trains the Random Forest model on the provided training data.
        predict(X_test): Makes predictions using the trained Random Forest model on the test data.
    """
    def __init__(self, n_estimators=100):
        """
        Initializes the Random Forest model with the specified number of trees.
        
        Parameters:
            n_estimators (int): The number of trees in the random forest (default is 100).
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)