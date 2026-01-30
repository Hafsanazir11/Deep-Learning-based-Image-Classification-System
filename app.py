import os
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template

#  Load Model 
MODEL_PATH = "models/resnet50_final.keras"
model = tf.keras.models.load_model(MODEL_PATH)

#  Flask App 
app = Flask(__name__)

# Ensure static directory exists
os.makedirs('static', exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("file")
        url = request.form.get("url")  # <-- new field for image URL

        img = None

        if file and file.filename != "":
            # Save uploaded file
            filepath = os.path.join("static", file.filename)
            file.save(filepath)
            img = image.load_img(filepath, target_size=(224, 224))
            image_path = filepath

        elif url and url.strip() != "":
            try:
                # Download image from URL
                response = requests.get(url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = img.resize((224, 224))

                # Save it temporarily
                filepath = os.path.join("static", "url_image.jpg")
                img.save(filepath)
                image_path = filepath
            except Exception as e:
                prediction = f"Error loading image from URL: {e}"

        # Run prediction only if img is available
        if img:
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

            pred = model.predict(img_array)[0][0]
            prediction = "Dog" if pred > 0.5 else "Cat"

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)