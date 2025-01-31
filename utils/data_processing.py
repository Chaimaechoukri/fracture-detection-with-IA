import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224

def preprocess_image(image_path):
    """
    Preprocess image for model prediction
    """
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Preprocess
    img = preprocess_input(img)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img
