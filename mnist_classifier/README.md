# MNIST Classifier Project

This project demonstrates the implementation of different machine learning algorithms to classify MNIST dataset images. The algorithms used in this project include:
- **Random Forest**
- **Feed-Forward Neural Network**
- **Convolutional Neural Network (CNN)**

The goal of this project is to provide a clear comparison of how each algorithm performs on the MNIST dataset, including training, prediction, and visualization of results.

## Project Structure

- **`models/`**: Contains the implementations of the classifiers.
    - `base.py`: Defines the interface `MnistClassifierInterface` that each classifier follows.
    - `random_forest.py`: Implements the `RandomForestMnistClassifier` using the Random Forest algorithm.
    - `feed_forward_nn.py`: Implements the `FeedForwardNNMnistClassifier` using a simple feed-forward neural network.
    - `cnn.py`: Implements the `CNNMnistClassifier` using a convolutional neural network.
- **`classifier.py`**: Contains the `MnistClassifier` class, which selects the appropriate model based on user input and wraps its training and prediction methods.
- **`notebook/`**: Jupyter Notebook demonstrating the models, with visualization of the results.
- **`requirements.txt`**: Lists all the dependencies needed to run the project.
- **`README.md`**: This file, explaining how to set up and use the project.

## Solution Explanation

This project trains and evaluates three different classifiers on the MNIST dataset:

1. **Random Forest**:
    - A tree-based model that creates multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees.

2. **Feed-Forward Neural Network**:
    - A basic neural network with a hidden layer. It uses the ReLU activation function for the hidden layer and softmax activation for the output layer.

3. **Convolutional Neural Network (CNN)**:
    - A deep learning model specialized for image classification. It uses convolutional layers followed by pooling and fully connected layers to predict the classes.

The `MnistClassifier` class is used to wrap each of these models. Depending on the algorithm specified (`'rf'`, `'nn'`, or `'cnn'`), the corresponding model is selected and trained.

The classifiers are trained using the MNIST training data and evaluated on the test data. Random noisy images are also tested to observe how the models handle noise.

### Results Visualization

The project includes functionality to visualize a random subset of test images, their true labels, and the predictions made by each classifier. Predictions are displayed in green if they are correct and red if they are incorrect.

## Requirements

To run the project, you need to install the dependencies listed in the `requirements.txt` file.

### Install Dependencies

pip install -r requirements.txt