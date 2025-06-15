from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(150, 150), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    val_datagen = ImageDataGenerator(rescale=1./255)  # Validasi tidak perlu di-augmentasi
    val_gen = val_datagen.flow_from_directory(
        f"{data_dir}/val",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    return train_gen, val_gen
