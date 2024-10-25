README
CNN Sign Language Recognition System
Description
This project aims to create a Convolutional Neural Network (CNN) model for recognizing sign language from audio signals represented as Mel Frequency Cepstral Coefficients (MFCC). The dataset consists of audio samples from a collection of spoken numbers, which are then converted into MFCC features and used to train the CNN model. The project integrates the Weights & Biases (wandb) tool for experiment tracking and performance visualization, making it easier to monitor the training progress and the model's accuracy.

Features
Data Loading and Preprocessing: Loads audio data, converts it into MFCC, and prepares it for training, validation, and testing.
CNN Model Building: Constructs a CNN model suitable for classifying the processed audio data.
Training and Validation: Utilizes the Keras library to train the model on the processed dataset, including callbacks for performance logging and model checkpointing.
Performance Evaluation: Evaluates the model on a test set and visualizes the accuracy and error over epochs. Additionally, performs a sample prediction and generates a confusion matrix for detailed performance analysis.
Prerequisites
Before you begin, ensure you have met the following requirements:

Python 3.10 installed
The following Python packages installed: os, json, numpy, sklearn, tensorflow, matplotlib, seaborn, inflect, librosa, pandas, tqdm, and wandb.
Dataset prepared in the specified directory structure and format.
Installation and Setup
Clone the project repository to your local machine.
Ensure all required Python packages are installed using pip install -r requirements.txt (assuming a requirements.txt file is provided and contains all necessary packages).
Set up a Weights & Biases account and project for tracking experiments.
Update the DATASET_PATH in the script to point to your dataset's location.
If necessary, adjust the labels and mappings to fit your dataset in the load_data function.
Usage
The project is divided into two main scripts:

CNN_Training.py: This script is responsible for loading the data, training the CNN model, and evaluating its performance.
MFCC_Extraction.py: Prior to training, this script extracts MFCC features from the audio dataset and saves them in a convenient format for the model.
To use this project:

Extract MFCC Features: Run MFCC_Extraction.py to process your dataset and extract MFCC features.


python MFCC_Extraction.py
Train and Evaluate the CNN Model: After extracting features, run CNN_Training.py to train the model and evaluate its performance.


python CNN_Training.py
Monitor training progress, view logs, and access model checkpoints using your Weights & Biases dashboard.

