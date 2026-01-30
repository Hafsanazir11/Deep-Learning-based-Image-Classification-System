# scripts/predict.py
import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

#Load model
model = tf.keras.models.load_model("models/resnet50_final.keras")

with open("models/class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping (0→Cat, 1→Dog)
idx_to_class = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    # Load & preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Prediction
    prob = model.predict(x, verbose=0)[0][0]
    pred_class = int(prob >= 0.5)  # threshold 0.5
    class_name = idx_to_class[pred_class]

    print(f"Image: {img_path}")
    print(f"Prediction: {class_name} ({prob:.4f} confidence)")
    return class_name, prob


if __name__ == "__main__":
   
    test_img = "test_images/cat1.jpg"  
    predict_image(test_img)
