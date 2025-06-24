# predict.py
import os
import json
import numpy as np
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# === PATH SETUP ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'plant_disease_model.h5')
CLASS_INDEX_PATH = os.path.join(BASE_DIR, 'model', 'class_indices.json')

IMG_HEIGHT, IMG_WIDTH = 224, 224
AUGMENTATIONS = 5

# === MODEL LOAD ===
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Make sure it's uploaded in the 'model/' folder.")

model = load_model(MODEL_PATH, compile=False)

# === LOAD CLASS LABELS ===
if not os.path.isfile(CLASS_INDEX_PATH):
    raise FileNotFoundError(f"❌ Class index file not found at {CLASS_INDEX_PATH}. Make sure it's uploaded correctly.")

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

# === IMAGE AUGMENTATION ===
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# === PREDICTION FUNCTION ===
def predict_image(image_path, n_augmentations=AUGMENTATIONS):
    """
    Predict the class of a plant leaf image using multiple augmentations.
    
    Args:
        image_path (str): Path to the image file.
        n_augmentations (int): Number of augmentations to average over.
    
    Returns:
        (str, float): Predicted class label and confidence percentage.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    predictions = []

    for _ in range(n_augmentations):
        augmented_img = next(datagen.flow(x, batch_size=1))
        pred = model.predict(augmented_img, verbose=0)
        predictions.append(pred)

    avg_pred = np.mean(predictions, axis=0)
    predicted_index = int(np.argmax(avg_pred))
    predicted_class = labels[predicted_index]
    confidence = float(np.max(avg_pred)) * 100

    print("\n🔍 Prediction Summary")
    print("=========================")
    print(f"🏷️  Predicted Label    : {predicted_class}")
    print(f"🔥 Confidence Score    : {confidence:.2f}%")
    print("=========================")

    return predicted_class, confidence

# === CLI SUPPORT ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_image")
        sys.exit(1)

    test_image_path = sys.argv[1]
    try:
        predict_image(test_image_path)
    except Exception as e:
        print("❌ Error:", str(e))
