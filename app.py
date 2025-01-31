from flask import Flask, request, render_template
import os
import tensorflow as tf
from utils.data_processing import preprocess_image

# Initialisation de Flask
app = Flask(__name__)

# Vérification et chargement du modèle
MODEL_PATH = "model/anomaly_detection_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file not found. Please run train.py first to train and save the model."
    )

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

# Définition des classes (Exemple : adapte cette liste selon ton modèle)
CLASS_NAMES = ["Pas de Fracture", "Fracture Simple", "Fracture Complexe"]

# Création du dossier pour les uploads
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Aucun fichier sélectionné"
    
    file = request.files['file']
    if file.filename == '':
        return "Aucun fichier sélectionné"

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Prétraiter l'image
    image = preprocess_image(file_path)

    # Faire une prédiction
    prediction = model.predict(image)
    predicted_index = prediction.argmax(axis=1)[0]
    predicted_class = CLASS_NAMES[predicted_index]

    return render_template("result.html", prediction=predicted_class, image=file_path)

if __name__ == "__main__":
    app.run(debug=True)
