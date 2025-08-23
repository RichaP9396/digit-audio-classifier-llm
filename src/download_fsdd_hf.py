import os
import requests
from zipfile import ZipFile

DATA_DIR = "data/fsdd"
os.makedirs(DATA_DIR, exist_ok=True)

FSDD_ZIP_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
ZIP_PATH = os.path.join(DATA_DIR, "fsdd.zip")

# Download the zip file
print("Downloading FSDD dataset...")
r = requests.get(FSDD_ZIP_URL, stream=True)
with open(ZIP_PATH, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
print("Download complete.")

# Extract all wav files
print("Extracting .wav files...")
with ZipFile(ZIP_PATH, "r") as zip_ref:
    for file in zip_ref.namelist():
        if file.endswith(".wav"):
            zip_ref.extract(file, DATA_DIR)

print("All WAV files extracted to", DATA_DIR)
