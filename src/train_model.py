# train_model.py
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# Paths
model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
model_dir = os.path.abspath(model_dir)

dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "color")
dataset_path = os.path.abspath(dataset_path)

model_save_path = os.path.join(model_dir, "plant_disease_model.keras")
class_index_path = os.path.join(model_dir, "class_indices.json")

# Create model directory
os.makedirs(model_dir, exist_ok=True)

# Image Preprocessing
img_height, img_width = 224, 224
batch_size = 32

datagen_train = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen_val = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen_train.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

val_gen = datagen_val.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

# Save class indices
with open(class_index_path, "w") as f:
    json.dump(train_gen.class_indices, f)

# Compute class weights
labels = train_gen.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights_dict = dict(enumerate(class_weights))

# Focal Loss
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=-1))
    return loss

# Model
base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(len(train_gen.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

# Checkpoint
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[checkpoint],
    class_weight=class_weights_dict
)

# Save training history
history_path = os.path.join(model_dir, "training_history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)

# Trigger evaluation
print("\n📊 Running Evaluation Script...")
os.system("python src/evaluate.py")
