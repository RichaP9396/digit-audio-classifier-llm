import torch
import torch.nn.functional as F
import numpy as np
import librosa
import sounddevice as sd
import time

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# Model definition (same as training)
# -----------------------
class DigitCNN(torch.nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2,2)
        dummy = torch.zeros(1,1,13,32)
        dummy = self.pool(torch.nn.functional.relu(self.conv2(torch.nn.functional.relu(self.conv1(dummy)))))
        flattened_size = dummy.numel()
        self.fc1 = torch.nn.Linear(flattened_size, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------
# Load trained model
# -----------------------
model = DigitCNN().to(device)
model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
model.eval()

# -----------------------
# Parameters
# -----------------------
SR = 8000       # Sampling rate
DURATION = 1.5  # seconds per recording (slightly longer)
N_MFCC = 13
N_FFT = 512

# -----------------------
# Preprocessing + Prediction
# -----------------------
def predict_digit_from_signal(signal, threshold=0.01):
    # Normalize audio
    signal = signal.astype(np.float32)
    signal = signal / (np.max(np.abs(signal)) + 1e-6)

    # Ignore quiet recordings
    if np.mean(np.abs(signal)) < threshold:
        return None  

    # Trim silence from beginning and end
    signal, _ = librosa.effects.trim(signal, top_db=25)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT)

    # Pad/trim to fixed length (32 frames)
    if mfcc.shape[1] < 32:
        mfcc = np.pad(mfcc, ((0,0),(0,32-mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :32]

    # Torch tensor
    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(F.softmax(output, dim=1), dim=1).item()
    return pred

# -----------------------
# Interactive Loop
# -----------------------
print("🎤 Press Enter, speak a digit (0-9), and wait for prediction.\n")
while True:
    try:
        input("Press Enter to record...")
        audio_data = sd.rec(int(SR*DURATION), samplerate=SR, channels=1)
        sd.wait()

        digit = predict_digit_from_signal(audio_data[:,0])
        if digit is not None:
            print(f"✅ Predicted digit: {digit}\n")
        else:
            print("⚠️ No valid speech detected, try again.\n")
    except KeyboardInterrupt:
        print("\nStopped listening.")
        break
