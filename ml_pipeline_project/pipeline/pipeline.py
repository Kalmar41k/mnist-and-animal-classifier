"""
This script defines a class `AnimalVerificationPipeline` that combines two inference models: 
1. A Named Entity Recognition (NER) model for identifying animal entities in text.
2. An image classification model for predicting the class of an image.

The class includes a method for verifying if the predicted animal label from an image matches any of the animal entities found in the provided sentence.

Key Features:
- Integrates two separate models: one for text (NER) and one for image classification.
- Verifies if an animal mentioned in the text matches the classification result from an image.
- Combines the results from both models to provide a boolean output.

Dependencies:
- `AnimalNER` for NER-based animal identification.
- `ImageClassifier` for image classification.
"""

from pipeline.ner_inference import AnimalNER
from pipeline.image_classification_inference import ImageClassifier

class AnimalVerificationPipeline:
    def __init__(self):
        """
        Initializes the AnimalVerificationPipeline class by loading the NER and image classification models.

        This class combines the outputs of the two models to verify if an animal mentioned in a sentence
        matches the classification of an image.

        :param ner_model: An instance of the AnimalNER model for NER-based animal identification.
        :param classifier: An instance of the ImageClassifier for image-based animal classification.
        """
        self.ner = AnimalNER()
        self.classifier = ImageClassifier()

    def verify_animal(self, sentence, img_path):
        """
        Verifies if the predicted animal label from the image matches any of the animals identified in the sentence.

        :param sentence: A sentence containing animal-related information.
        :param img_path: Path to the image to be classified.
        :return: A boolean value indicating whether the predicted animal in the image matches any animals in the sentence.
        """
        animals = self.ner.predict_animals(sentence)
        predicted_label = self.classifier.predict(img_path)

        return predicted_label in animals