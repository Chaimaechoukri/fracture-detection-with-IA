from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import pandas as pd

app = Flask(__name__)

# Load the model and class names globally
print("Loading model...")
model = tf.keras.models.load_model('image_classifier_model.h5')

# Load class names from test CSV
test_csv_path = os.path.join('data/test', '_classes.csv')
df = pd.read_csv(test_csv_path)
CLASS_NAMES = df.columns[1:].tolist()

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Save the uploaded file temporarily
        temp_path = 'temp_upload.jpg'
        file.save(temp_path)
        
        # Preprocess the image
        img_array = preprocess_image(temp_path)
        
        # Make prediction
        predictions = model.predict(img_array)[0]
        
        # Format results
        results = []
        for class_name, probability in zip(CLASS_NAMES, predictions):
            if probability > 0.5:  # Only include predictions with >50% confidence
                results.append({
                    'class': class_name,
                    'probability': float(probability) * 100
                })
        
        # Sort results by probability
        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 