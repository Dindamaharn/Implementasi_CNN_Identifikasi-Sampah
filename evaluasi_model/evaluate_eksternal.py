# evaluasi_model/evaluate_eksternal.py

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.preprocessing import image


# Dapatkan path absolut ke folder project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Path ke folder datatest_eksternal
test_dir = os.path.join(BASE_DIR, 'dataset', 'datatest_eksternal')

# Path ke model
model_paths = {
    "Before Optimasi": os.path.join(BASE_DIR, "models", "model_awal.h5"),
    "After Optimasi": os.path.join(BASE_DIR, "models", "model_optimasi.h5")
}

# Image size dan batch size
img_size = (150, 150)
batch_size = 32

# Preprocessing untuk data test
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Evaluasi kedua model
for label, path in model_paths.items():
    model = load_model(path)
    loss, accuracy = model.evaluate(test_generator, verbose=0)
    print(f"Akurasi {label}: {accuracy*100:.2f}%")
