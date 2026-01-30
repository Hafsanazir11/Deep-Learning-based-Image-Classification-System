import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

# Source directories
cat_dir = "data/Cat"
dog_dir = "data/Dog"

# Create split directories
os.makedirs("data/splits/train/cat", exist_ok=True)
os.makedirs("data/splits/train/dog", exist_ok=True)
os.makedirs("data/splits/val/cat", exist_ok=True)
os.makedirs("data/splits/val/dog", exist_ok=True)
os.makedirs("data/splits/test/cat", exist_ok=True)
os.makedirs("data/splits/test/dog", exist_ok=True)

# Helper function to split images
def split_images(source_dir, split_name):
    images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    train, temp = train_test_split(images, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    for img in train:
        shutil.copy(os.path.join(source_dir, img), f"data/splits/train/{split_name}/{img}")
    for img in val:
        shutil.copy(os.path.join(source_dir, img), f"data/splits/val/{split_name}/{img}")
    for img in test:
        shutil.copy(os.path.join(source_dir, img), f"data/splits/test/{split_name}/{img}")

split_images(cat_dir, "cat")
split_images(dog_dir, "dog")

print("✓ Data prepared successfully!")