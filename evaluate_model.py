import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load your trained model
model = tf.keras.models.load_model("models/resnet50_final.keras")

# Set up test data
test_dir = "data/splits/test"
img_size = (224, 224)
batch_size = 32

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode="binary", shuffle=False
)

# Get predictions
print("\n=== Model Evaluation ===\n")
probs = model.predict(test_gen, verbose=0).ravel()
y_pred = (probs >= 0.5).astype(int)
y_true = test_gen.classes

# Test Accuracy
loss, accuracy = model.evaluate(test_gen, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}\n")

# Classification Report (Precision, Recall, F1-Score)
report = classification_report(y_true, y_pred, target_names=["Cat", "Dog"])
print("Classification Report:")
print(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save to file
with open("model_evaluation.txt", "w") as f:
    f.write("=== Model Evaluation ===\n\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))

print("\n✓ Saved to model_evaluation.txt")