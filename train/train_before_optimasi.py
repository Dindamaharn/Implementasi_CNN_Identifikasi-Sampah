# train/train_before_optimasi.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from preprocessing_data.data_loader import load_data

train_gen, val_gen = load_data("dataset/datasplit", batch_size=32)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_gen, epochs=20, steps_per_epoch=25, validation_data=val_gen)

model.save("models/model_awal.h5")

# Simpan grafik
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Akurasi - Sebelum Optimasi")
plt.savefig("results/plots/before_optimasi/accuracy.png")
plt.clf()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss - Sebelum Optimasi")
plt.savefig("results/plots/before_optimasi/loss.png")

import pickle
with open("models/history_before_optimasi.pkl", "wb") as f:
    pickle.dump(history.history, f)
    
print("✅ Training model awal selesai!")
