import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Parameters
IMAGE_SIZE = (64, 64)
DATASET_PATH = 'dataset'

def load_data(dataset_path):
    images = []
    labels = []
    
    for label in os.listdir(dataset_path):
        for img_file in os.listdir(os.path.join(dataset_path, label)):
            img_path = os.path.join(dataset_path, label, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMAGE_SIZE)
            images.append(img)
            labels.append(label)
    
    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)
    
    return images, labels

images, labels = load_data(DATASET_PATH)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# Save the classes
np.save('classes.npy', label_encoder.classes_)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, y_encoded, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
model.save('kannada_sign_language_model.h5')
