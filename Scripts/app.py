import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"


SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
LOCAL_MODEL_PATH = "D:/Projects/IMG_Upscale/Model"

# Create the directory if it doesn't exist
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

# Load the model from the hub
model = hub.load(SAVED_MODEL_PATH)

# Save the model locally
tf.saved_model.save(model, LOCAL_MODEL_PATH)

print(f"Model saved locally at {LOCAL_MODEL_PATH}")