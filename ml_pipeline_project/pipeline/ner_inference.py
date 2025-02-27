"""
This script defines a class `AnimalNER` that utilizes a pretrained DistilBERT model for Named Entity Recognition (NER)
to identify animal entities in a given sentence. The class includes methods to load the pretrained model and tokenizer,
as well as to perform inference on a sentence to extract animal entities.

Key Features:
- Loads a pretrained DistilBERT model fine-tuned for NER tasks with a focus on animal identification.
- Tokenizes input sentences and performs inference to classify tokens as either 'ANIMAL' or 'O' (non-entity).
- Outputs a list of tokens identified as animals in the sentence.

Dependencies:
- `transformers` library for the DistilBERT model and tokenizer.
- `torch` for handling model inference.
"""

from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
import torch

class AnimalNER:
    def __init__(self, model_path="models/ner_model"):
        """
        Loads a trained Named Entity Recognition (NER) model and its corresponding tokenizer.

        :param model_path: Path to the directory containing the pretrained model and tokenizer (default is "models/ner_model").
        """
        self.model = DistilBertForTokenClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.labels = ['O', 'ANIMAL']
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def predict_animals(self, sentence):
        """
        Takes a sentence and returns a list of recognized animal entities.

        :param sentence: The text to be analyzed for animal entities.
        :return: A list of recognized animal names found in the sentence.
        """
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_labels = [self.id2label[id.item()] for id in predicted_ids[0]]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        animals = [token for token, label in zip(tokens, predicted_labels) if label == "ANIMAL"]
        return animals