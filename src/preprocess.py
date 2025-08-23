import os
import librosa
import numpy as np
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Folder containing .wav files
DATA_DIR = "data/fsdd"
SAVE_PATH = "data/fsdd_features.npz"

# Parameters
SR = 8000       # Sampling rate
N_MFCC = 13     # Number of MFCC features
N_FFT = 512     # Smaller FFT window to avoid warning

X = []
y = []

# Iterate over all wav files
print("Extracting MFCC features from audio files...")
for file in os.listdir(DATA_DIR):
    if file.endswith(".wav"):
        label = int(file.split("_")[0])  # label is the first number in filename
        filepath = os.path.join(DATA_DIR, file)
        signal, sr = librosa.load(filepath, sr=SR)
        # Use keyword y= and smaller n_fft
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT)
        # Normalize length to same size (optional, pad/truncate)
        if mfcc.shape[1] < 32:
            mfcc = np.pad(mfcc, ((0,0), (0, 32 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :32]
        X.append(mfcc)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Extracted {len(X)} samples with shape {X[0].shape}")
print("Saving features and labels to", SAVE_PATH)

np.savez(SAVE_PATH, X=X, y=y)
print("Preprocessing complete!")
