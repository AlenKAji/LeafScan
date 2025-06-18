# evaluate.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
model_dir = "C:/Users/Win11/Desktop/projects/Plant/model"
dataset_path = "C:/Users/Win11/Desktop/projects/Plant/dataset/color/"

model_path = os.path.join(model_dir, "plant_disease_model.keras")
class_index_path = os.path.join(model_dir, "class_indices.json")
history_path = os.path.join(model_dir, "training_history.json")
report_path = os.path.join(model_dir, "classification_report.txt")
conf_matrix_path = os.path.join(model_dir, "confusion_matrix.png")
plot_path = os.path.join(model_dir, "training_plot.png")

# Load class indices
with open(class_index_path, "r") as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

# Image data
img_height, img_width = 224, 224
batch_size = 32

datagen_val = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_gen = datagen_val.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

# Load model
model = tf.keras.models.load_model(model_path, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Predict
Y_true = val_gen.classes
Y_pred_probs = model.predict(val_gen)
Y_pred = np.argmax(Y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(conf_matrix_path)
plt.close()

# Classification report
report = classification_report(Y_true, Y_pred, target_names=class_names)
with open(report_path, "w") as f:
    f.write(report)

# Plot training history
if os.path.exists(history_path):
    with open(history_path, "r") as f:
        history = json.load(f)

    acc = history.get('accuracy', [])
    val_acc = history.get('val_accuracy', [])
    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])
    epochs_range = range(len(acc))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

print("✅ Evaluation artifacts saved to:", model_dir)
