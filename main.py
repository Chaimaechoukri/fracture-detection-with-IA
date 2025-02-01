import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
import os

class DataGenerator(Sequence):
    def __init__(self, csv_path, img_dir, batch_size=32, img_size=(224, 224), shuffle=True):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.class_columns = self.df.columns[1:].tolist()
        self.n = len(self.df)
        self.indexes = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.n)
        batch_indexes = self.indexes[start_idx:end_idx]

        batch_x = []
        batch_y = []

        for i in batch_indexes:
            row = self.df.iloc[i]
            try:
                img_path = os.path.join(self.img_dir, row['filename'])
                if os.path.exists(img_path):
                    img = load_img(img_path, target_size=self.img_size)
                    img_array = img_to_array(img)
                    img_array = img_array / 255.0
                    batch_x.append(img_array)
                    label = row[self.class_columns].values.astype('float32')
                    batch_y.append(label)
            except Exception as e:
                print(f"Error loading image {row['filename']}: {str(e)}")
                continue

        if len(batch_x) == 0:
            # If no valid images in batch, recursively try next batch
            return self.__getitem__((idx + 1) % len(self))

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def create_model(num_classes, input_shape=(224, 224, 3)):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])
    return model

# Main execution
if __name__ == "__main__":
    try:
        # Define data directories and parameters
        train_dir = 'data/train'
        valid_dir = 'data/valid'
        test_dir = 'data/test'
        batch_size = 16  # Reduced batch size for better memory management
        
        print("Creating data generators...")
        # Create data generators
        train_generator = DataGenerator(
            os.path.join(train_dir, '_classes.csv'),  # Fixed CSV filename
            train_dir,
            batch_size=batch_size
        )
        
        valid_generator = DataGenerator(
            os.path.join(valid_dir, '_classes.csv'),  # Fixed CSV filename
            valid_dir,
            batch_size=batch_size,
            shuffle=False
        )
        
        test_generator = DataGenerator(
            os.path.join(test_dir, '_classes.csv'),  # Fixed CSV filename
            test_dir,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Get number of classes from the first batch
        print("Loading first batch to determine number of classes...")
        _, first_batch_y = train_generator[0]
        num_classes = first_batch_y.shape[1]
        print(f"Number of classes: {num_classes}")
        
        # Create and compile model
        print("Creating and compiling model...")
        model = create_model(num_classes)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        # Train model
        print("Starting training...")
        history = model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=20,
            verbose=1
        )
        
        # Evaluate model
        print("Evaluating model...")
        test_results = model.evaluate(test_generator)
        print(f"\nTest loss, Test accuracy:", test_results)
        
        # Save model
        print("Saving model...")
        model.save('image_classifier_model.h5')
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
