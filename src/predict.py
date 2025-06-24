# predict.py
import json
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import sys

# === CONFIGURATION ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "plant_disease_model.keras")
MODEL_PATH = os.path.abspath(MODEL_PATH)
CLASS_INDEX_PATH  = os.path.join(os.path.dirname(__file__), "..", "model", "class_indices.json")
CLASS_INDEX_PATH  = os.path.abspath(CLASS_INDEX_PATH )


IMG_HEIGHT, IMG_WIDTH = 224, 224
AUGMENTATIONS = 5  # Number of augmented predictions

# === LOAD MODEL ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)


# === LOAD CLASS INDICES AND MAP TO LABELS ===
if not os.path.exists(CLASS_INDEX_PATH):
    raise FileNotFoundError(f"❌ Class index file not found at {CLASS_INDEX_PATH}")
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}  # Reversed for index-to-label

# === IMAGE AUGMENTATION GENERATOR ===
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
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    predictions = []

    for _ in range(n_augmentations):
        augmented_iter = datagen.flow(x, batch_size=1)
        augmented_img = next(augmented_iter)
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

# === COMMAND LINE USAGE ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_image")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        predict_image(image_path)
    except Exception as e:
        print("❌ Error:", str(e))
