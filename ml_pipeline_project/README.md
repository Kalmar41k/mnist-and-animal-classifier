## Animal Verification Pipeline

### Overview

This project implements an Animal Verification Pipeline that combines two machine learning models: one for Named Entity Recognition (NER) and one for Image Classification. The goal is to verify if an animal mentioned in a sentence matches the animal depicted in an image. This is achieved using two separate models, one based on the transformer architecture (DistilBERT) for NER, and a fine-tuned VGG16 model for image classification.

The system processes a text input (e.g., "There is a dog in the picture.") and an image input, and outputs a boolean value indicating whether the predicted animal in the image matches the animal extracted from the text.

### Project Structure

- data: Contains datasets for training the models:
    - ner_dataset.json: A dataset of sentences for training the NER model (transformers-based).
    - animals-10: An image dataset of 10 animal classes (butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel). This dataset should be downloaded from Kaggle (instructions provided below).
- exploratory_data_analysis:
    - animals10_eda.ipynb: Jupyter Notebook for exploratory data analysis (EDA) of the animals-10 dataset. The analysis suggests that data augmentation is needed due to the uneven distribution of classes.
- models:
    - ner_model: The DistilBERT-based NER model with an accuracy of 82%.
    - vgg16_finetuned.h5: The fine-tuned VGG16 image classification model with an accuracy of 87%.
- pipeline:
    - Contains classes for NER (ner_inference.py), image classification (image_classification_inference.py), and the final pipeline (pipeline.py) that combines the predictions from both models.
- train:
    - Scripts for training the models:
        - train_image_classification.py: Script to process data, train, and save the image classification model.
        - train_ner.py: Script to train and save the NER model.
    - These scripts require the animals-10 dataset, which must be downloaded from Kaggle (instructions below).
- demo.ipynb:
    - A Jupyter Notebook that demonstrates the functionality of the pipeline using the classes in the pipeline folder. This notebook should be run to interact with the models and verify the animal in the image based on the input text.
- requirements.txt:
    - A text file listing all the dependencies used in the project. Install them using pip install -r requirements.txt.

### Requirements:

#### General Requirements
- Python 3
- All dependencies listed in requirements.txt
- A Kaggle token for downloading the animals-10 dataset (explained below)

#### Installing Dependencies
1. Clone this repository.
2. Install the required Python libraries using the following command:
    ```bash
    pip install -r requirements.txt
    ```
3. If you don't have the Kaggle dataset, you'll need a Kaggle API token.

#### Kaggle Token Setup
To download the animals-10 dataset from Kaggle, you need to set up a Kaggle API token:
1. Go to Kaggle and log in.
2. Navigate to your account settings (click on your profile icon in the top right corner).
3. Scroll down to the "API" section and click "Create New API Token".
4. This will download a file called kaggle.json.
5. Place kaggle.json in the directory ~/.kaggle/ (for Linux or macOS) or C:\Users\<YourName>\.kaggle\ (for Windows).

After setting up the token, you can download dataset in the code.

### Running the Project
1. To run the demo: Open Demo.ipynb and follow the instructions to test the models and pipeline.
2. To retrain the models: Use the scripts in the train folder:
    - Run train_image_classification.py to train the image classification model.
    - Run train_ner.py to train the NER model.

### How It Works:

#### Run the Demo
Once the models are trained, open demo.ipynb and provide both a text input (e.g., "There is a dog in the picture.") and an image. The pipeline will use both models to verify if the animal mentioned in the sentence matches the animal in the image.
#### Example:
```bash
text = "There is a dog in the picture."
img_path = "path_to_image.jpg"

result = pipeline.verify_animal(text, img_path)
print(result)  # True if the animal in the image matches the one in the text, else False
```

#### Train models
If you want to re-train these models, run the scripts:
- train_image_classification.py to train a model that recognizes animals in images.
- train_ner.py to train a model that recognizes animals in text.

### Results:
- NER Model: Achieves an accuracy of 82% on the animal entity recognition task.
- Image Classification Model: Achieves an accuracy of 87% on classifying animals from the animals-10 dataset.
- Pipeline: Combines the results from both models to output a boolean value indicating if the animal in the text matches the animal in the image.