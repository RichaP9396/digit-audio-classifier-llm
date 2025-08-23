# Spoken Digit Recognition using PyTorch

This repository contains a lightweight prototype for recognizing spoken digits (0–9) from audio recordings using PyTorch. The project leverages the Free Spoken Digit Dataset (FSDD) and a simple CNN architecture to predict digits from MFCC features.


## Project Structure

├── data/ # Downloaded audio files
├── src/
│ ├── model.py # CNN model definition
│ ├── preprocess.py # Feature extraction (MFCC)
│ ├── train.py # Training loop
│ ├── inference.py # Predict from .wav file
│ ├── mic_inference.py # Real-time prediction using microphone
| ├── download_fsdd_hf.py # Script to download FSDD dataset
├── requirements.txt # Dependencies
├── venv # python environment
└── README.md # This file


## Setup

Setup & Installation

# 1. Clone the repository
bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

Create & activate virtual environment

Windows (PowerShell):

python -m venv venv
.\venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Download the dataset
python download_fsdd_hf.py

This will create a data/fsdd folder with WAV files.
After extraction, check the folder: dir .\data\fsdd
->Move all .wav files to data/fsdd
# Go to the extracted folder
cd .\data\fsdd\free-spoken-digit-dataset-master

# Move all wav files to the parent fsdd folder
Get-ChildItem -Recurse -Filter "*.wav" | Move-Item -Destination ..\

# Go back to main folder
cd ..\..\..
# Clean up extra folders and ZIP (optional)
# Remove the extracted folder and ZIP to save space
Remove-Item -Recurse -Force ".\data\fsdd\free-spoken-digit-dataset-master"
Remove-Item -Force ".\data\fsdd\fsdd.zip"
# Verify
dir .\data\fsdd


You should now see all .wav files directly in data/fsdd, like:

0_jackson_0.wav
1_nicolas_1.wav
2_theo_3.wav
...


# Feature Extraction

We extract MFCC features using librosa from each audio file. The features are saved as .npz for efficient training.

python src/preprocess.py

# Training

Train the CNN model:

python src/train.py


Default epochs: 20

Batch size: 32

Learning rate: 0.001

The trained model is saved as digit_cnn.pth.

# Evaluation

Validate the model on held-out data:

python src/inference.py


Example usage:

Enter path to .wav file: data/fsdd/0_george_0.wav
Predicted digit: 0

# Real-Time Microphone Prediction
python src/mic_inference.py


Press Enter to record audio.

Speak a digit (0–9).

The model predicts and displays the digit.

Results

Model: CNN on MFCC features

Dataset: Free Spoken Digit Dataset (3000 samples)

Validation Accuracy: 95–97%

# Requirements

Python 3.8+

PyTorch

Librosa

NumPy

Matplotlib

Scikit-learn

Sounddevice (for microphone input)

Install all dependencies:

# pip install -r requirements.txt
