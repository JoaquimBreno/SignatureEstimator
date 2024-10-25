# CNN Sign Language Recognition System

## Description
This project is focused on building a **Convolutional Neural Network (CNN)** to recognize sign language from audio signals by utilizing **Mel Frequency Cepstral Coefficients (MFCC)** as feature inputs. Audio samples of spoken numbers are converted into MFCC features, which are used to train the CNN model. The project also incorporates **Weights & Biases (wandb)** for tracking experiments, providing easy monitoring of training progress and model accuracy.

## Features

- **Data Loading and Preprocessing**  
  Loads audio data, converts it into MFCC features, and prepares it for training, validation, and testing phases.

- **CNN Model Building**  
  Constructs a CNN model designed for classifying the processed audio data.

- **Training and Validation**  
  Uses the Keras library to train the model on the dataset, with callbacks for performance logging and model checkpointing.

- **Performance Evaluation**  
  Evaluates the model on a test set, visualizes accuracy and error across epochs, and generates a confusion matrix for detailed performance analysis. Additionally, performs a sample prediction for quick assessment.

---

## Prerequisites
Ensure the following requirements are met before starting:

1. **Python 3.10** installed.
2. The following Python packages installed:
   - `os`, `json`, `numpy`, `sklearn`, `tensorflow`, `matplotlib`, `seaborn`, `inflect`, `librosa`, `pandas`, `tqdm`, `wandb`
3. **Dataset** prepared in the specified directory structure and format.

---

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
Install the required packages: Ensure all necessary packages are installed:

```bash
    Copiar código
    pip install -r requirements.txt
    Set up Weights & Biases (wandb):
```

Create an account at Weights & Biases (if you don't have one).
Configure a project for tracking your experiments.
Update Dataset Path:

Update the DATASET_PATH in the script to point to your dataset’s location.
If necessary, customize label mappings in the load_data function to align with your dataset.
Usage
The project consists of two main scripts:

1. Extract MFCC Features
Run the mfcc_extraction.py script to process your dataset and extract MFCC features:

```bash
Copiar código
python mfcc_extraction.py
```
2. Train and Evaluate the Model
After feature extraction, use the cnn_train.py script to train the model and evaluate its performance:

```bash
Copiar código
python cnn_train.py
```
Note: Monitor training progress, logs, and model checkpoints via your Weights & Biases dashboard.

## DATASETS

Meter2800(FMA,MAG,OWN)
https://www.sciencedirect.com/science/article/pii/S2352340923008053?ref=pdf_download&fr=RR-2&rr=8d844b073a2f27e7
GTZAN
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
