import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def load_and_preprocess_image(image_path, img_size=(224, 224)):
    """Load and preprocess a single image"""
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_image(model, image_path, class_names):
    """Predict classes for a single image"""
    # Preprocess the image
    img_array = load_and_preprocess_image(image_path)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get predictions
    predictions = model.predict(img_array)
    
    # Create a dictionary of class predictions
    results = {}
    for class_name, pred_value in zip(class_names, predictions[0]):
        results[class_name] = float(pred_value)
    
    return results

def main():
    try:
        # Load the trained model
        print("Loading model...")
        model = tf.keras.models.load_model('image_classifier_model.h5')
        
        # Load class names from test CSV
        test_csv_path = os.path.join('data/test', '_classes.csv')
        df = pd.read_csv(test_csv_path)
        class_names = df.columns[1:].tolist()  # Skip filename column
        
        # Test directory
        test_dir = 'data/test'
        
        # Get list of test images
        test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print("\nStarting predictions...")
        print("-" * 50)
        
        # Process each test image
        for image_name in test_images[:5]:  # Test first 5 images
            image_path = os.path.join(test_dir, image_name)
            print(f"\nProcessing image: {image_name}")
            
            # Get predictions
            predictions = predict_image(model, image_path, class_names)
            
            # Display results
            print("\nPredictions:")
            print("-" * 20)
            for class_name, probability in predictions.items():
                if probability > 0.5:  # Show only classes with probability > 50%
                    print(f"{class_name}: {probability:.2%}")
            
            # Get ground truth from CSV
            true_labels = df[df['filename'] == image_name][class_names].iloc[0]
            
            print("\nGround Truth:")
            print("-" * 20)
            for class_name, value in true_labels.items():
                if value > 0:
                    print(f"{class_name}: {value}")
            
            print("\n" + "="*50)
        
        print("\nTesting completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
