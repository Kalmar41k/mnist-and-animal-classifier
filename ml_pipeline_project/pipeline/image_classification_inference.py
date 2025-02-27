"""
This script defines a class `ImageClassifier` that loads a pretrained image classification model (VGG16 fine-tuned),
processes input images, and predicts the class of the given image. The class also includes methods for image preprocessing
and classification using the model.

Key Features:
- Loads a pretrained image classification model from a specified path.
- Loads the class labels from a file to map model outputs to human-readable labels.
- Preprocesses input images to match the input format required by the model (resizing, scaling).
- Predicts the class of a given image by performing inference with the model.
  
Dependencies:
- `tensorflow` for model loading and image preprocessing.
- `numpy` for numerical operations (e.g., processing image arrays).
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

class ImageClassifier:
    def __init__(self, model_path="models/vgg16_finetuned.h5", labels_path="data/class_labels.txt"):
        """
        Initializes the ImageClassifier class by loading the pretrained model and the class labels.

        :param model_path: Path to the pretrained model (default is "models/vgg16_finetuned.h5").
        :param labels_path: Path to the file containing class labels (default is "data/class_labels.txt").
        """
        self.model = load_model(model_path)
        with open(labels_path, "r") as f:
            self.class_labels = [label.strip() for label in f.readlines()]

    def preprocess_image(self, img_path):
        """
        Preprocesses the input image to the format required by the model (resizing and scaling).

        :param img_path: Path to the image file to be preprocessed.
        :return: A preprocessed image array ready for model prediction.
        """
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def predict(self, img_path):
        """
        Predicts the class of the input image by passing it through the model.

        :param img_path: Path to the image file to be classified.
        :return: The predicted class label for the input image.
        """
        img_array = self.preprocess_image(img_path)
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions)
        return self.class_labels[predicted_class]