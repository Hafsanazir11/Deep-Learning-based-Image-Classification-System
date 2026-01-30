import os
from PIL import Image

def clean_corrupted_images(data_dir):
    """Remove corrupted image files"""
    removed_count = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                img = Image.open(filepath)
                img.verify()  # Verify image is valid
            except Exception as e:
                print(f"❌ Removing corrupted image: {filepath}")
                os.remove(filepath)
                removed_count += 1
    
    print(f"✓ Removed {removed_count} corrupted images")

# Clean the data
clean_corrupted_images("data/splits")
print("✓ Image cleaning complete!")