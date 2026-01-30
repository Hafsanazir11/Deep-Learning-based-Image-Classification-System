import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


base_dir = "data/splits"  
train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "val")
test_dir  = os.path.join(base_dir, "test")

img_size = (224, 224)  
batch_size = 32
epochs_stage1 = 10     
epochs_stage2 = 5       
os.makedirs("models", exist_ok=True)

#  Data Generators 
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,        # Random rotation
    width_shift_range=0.2,    # Horizontal shift
    height_shift_range=0.2,   # Vertical shift
    zoom_range=0.3,           # Zoom
    shear_range=0.2,          # Shear
    horizontal_flip=True,     # Flip images horizontally
    brightness_range=(0.8, 1.2), # Brightness adjust
    fill_mode="nearest"
)

val_datagen  = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode="binary"
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode="binary"
)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode="binary", shuffle=False
)

# Save label mapping (for prediction later)
with open("models/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

# Model (ResNet50)
base = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base.trainable = False 

inputs = layers.Input(shape=(224, 224, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)  
model = models.Model(inputs, outputs)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Callbacks for early stopping and saving the best model
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_loss", verbose=1),
    ModelCheckpoint("models/resnet50_feature.keras", save_best_only=True, monitor="val_accuracy")
]

#  Stage 1: Feature Extraction 
print("\n=== Stage 1: Feature extraction (frozen base) ===")
history1 = model.fit(train_gen, validation_data=val_gen, epochs=epochs_stage1, callbacks=callbacks)

print("\nEvaluating on test set (Stage 1 model)...")
loss1, acc1 = model.evaluate(test_gen, verbose=0)
print(f"Stage 1 → Test Accuracy: {acc1:.2f}")

# Classification report (Stage 1)
probs1 = model.predict(test_gen, verbose=0).ravel()
y_pred1 = (probs1 >= 0.5).astype(int)
y_true = test_gen.classes
print("\nClassification Report (Stage 1):")
print(classification_report(y_true, y_pred1, target_names=["Cat","Dog"]))
print("Confusion Matrix (Stage 1):")
print(confusion_matrix(y_true, y_pred1))

#  Stage 2: Fine-Tuning 
print("\n=== Stage 2: Fine-tuning top layers ===")
base.trainable = True

# Freeze all layers except for the last 50 layers
for layer in base.layers[:-50]:
    layer.trainable = False

# Recompile the model with a smaller learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy",
              metrics=["accuracy"])

callbacks_ft = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
    ReduceLROnPlateau(patience=2, factor=0.2, monitor="val_loss", verbose=1),
    ModelCheckpoint("models/resnet50_finetuned.keras", save_best_only=True, monitor="val_accuracy")
]

# Train the model (fine-tuning)
history2 = model.fit(train_gen, validation_data=val_gen, epochs=epochs_stage2, callbacks=callbacks_ft)

print("\nEvaluating on test set (Fine-tuned model)...")
loss2, acc2 = model.evaluate(test_gen, verbose=0)
print(f"Fine-tuned → Test Accuracy: {acc2:.2f}")

# Final save
model.save("models/resnet50_final.keras")
print("Saved: models/resnet50_feature.keras, models/resnet50_finetuned.keras, models/resnet50_final.keras")
