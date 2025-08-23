# inference.py
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import os
from train import DigitCNN  # Import your model class


# Device setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load trained model

model = DigitCNN().to(device)
model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
model.eval()


# Audio preprocessing parameters

SR = 8000
N_MFCC = 13
N_FFT = 512
FRAME_LENGTH = 32


# Prediction function

def predict_digit(filepath):
    signal, sr = librosa.load(filepath, sr=SR)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT)
    if mfcc.shape[1] < FRAME_LENGTH:
        mfcc = np.pad(mfcc, ((0,0),(0, FRAME_LENGTH - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :FRAME_LENGTH]
    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(F.softmax(output, dim=1), dim=1).item()
    return pred


# Recursive search for all WAV files

def list_wav_files(base_folder="data/fsdd"):
    files = []
    for root, dirs, filenames in os.walk(base_folder):
        for f in filenames:
            if f.endswith(".wav"):
                files.append(os.path.join(root, f))
    if not files:
        print("No WAV files found in", base_folder)
        return []
    print("\nFound WAV files:")
    for i, f in enumerate(files):
        print(f"{i}: {os.path.basename(f)}")
    return files


# Main interactive loop

if __name__ == "__main__":
    files = list_wav_files()
    if not files:
        exit()

    while True:
        try:
            choice = input(f"\nEnter the number or filename to predict, or 'q' to quit: ")
            if choice.lower() == 'q':
                print("Exiting...")
                break

            # If input is a number (index)
            if choice.isdigit():
                idx = int(choice)
                if idx < 0 or idx >= len(files):
                    print("Invalid number. Try again.")
                    continue
                file_path = files[idx]
            else:
                # Match input as filename
                matches = [f for f in files if os.path.basename(f) == choice]
                if not matches:
                    print("Filename not found. Try again.")
                    continue
                file_path = matches[0]

            digit = predict_digit(file_path)
            print(f"Predicted digit for '{os.path.basename(file_path)}': {digit}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print("Error:", e)
