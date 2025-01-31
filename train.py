import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model
import pandas as pd
import numpy as np
import os

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

class MultiLabelGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, csv_file, batch_size=32, img_size=(224, 224), shuffle=True):
        self.directory = directory
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        
        # Read CSV and get labels
        self.df = pd.read_csv(os.path.join(directory, csv_file))
        self.image_filenames = self.df['filename'].values
        # Get all columns except 'filename' and 'fracture' as they are labels
        self.labels = self.df.drop(['filename', 'fracture'], axis=1).values
        
        self.n = len(self.image_filenames)
        self.indexes = np.arange(self.n)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Generate batch data
        batch_x = np.zeros((len(indexes), *self.img_size, 3))
        batch_y = np.zeros((len(indexes), self.labels.shape[1]))
        
        for i, index in enumerate(indexes):
            # Load and preprocess image
            img_path = os.path.join(self.directory, self.image_filenames[index])
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img / 255.0  # Normalize
            
            batch_x[i] = img
            batch_y[i] = self.labels[index]
        
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Create data generators
train_generator = MultiLabelGenerator('data/train', '_classes.csv', BATCH_SIZE, (IMG_SIZE, IMG_SIZE))
valid_generator = MultiLabelGenerator('data/valid', '_classes.csv', BATCH_SIZE, (IMG_SIZE, IMG_SIZE))

# Create model
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze the base model
base_model.trainable = False

# Create new model on top
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(13, activation='sigmoid')(x)  # 13 classes (excluding 'filename' and 'fracture')
model = Model(inputs, outputs)

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
print("Starting training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator
)

# Save model
print("Saving model...")
model.save('model/anomaly_detection_model.h5')
print("Model saved successfully!") 