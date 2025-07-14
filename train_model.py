import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# === 1. Path dan Parameter ===
CSV_PATH = "dataset/HAM10000_metadata.csv"
IMG_DIR = "all_image"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
USE_ALL_DATA = True  # set True agar tidak dibatasi LIMIT

# === 2. Load metadata dan label ===
df = pd.read_csv(CSV_PATH)
df = df[df['dx'].isin(['melanoma', 'bkl', 'nv'])].reset_index(drop=True)
label_mapping = {'melanoma': 0, 'bkl': 1, 'nv': 2}
df['label'] = df['dx'].map(label_mapping)

available_ids = {f.replace(".jpg", "") for f in os.listdir(IMG_DIR) if f.endswith(".jpg")}
df = df[df['image_id'].isin(available_ids)].reset_index(drop=True)

# === 3. Load gambar ===
images, labels = [], []
for i in range(len(df) if USE_ALL_DATA else min(len(df), 8000)):
    try:
        img_path = os.path.join(IMG_DIR, df['image_id'][i] + ".jpg")
        img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
        images.append(np.array(img) / 255.0)
        labels.append(df['label'][i])
    except:
        continue

X = np.array(images, dtype=np.float32)
y = to_categorical(labels, num_classes=3)

# === 4. Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
y_train_labels = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights = dict(enumerate(class_weights))

# === 5. Augmentasi Data ===
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
train_gen = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)


base_model = MobileNetV2(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
base_model.trainable = False  # freeze dulu

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, verbose=1)

# === 8. Training tahap 1 ===
model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# === 9. Fine-tuning tahap 2 (unfreeze sebagian) ===
for layer in base_model.layers[-30:]:  # hanya unfreeze 30 layer terakhir
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_gen,
    epochs=10,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# === 10. Evaluasi akhir ===
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Akurasi Test: {acc * 100:.2f}%")

# === 11. Simpan model ===
os.makedirs("model", exist_ok=True)
model.save("model/skin_cancer2.h5")
print("✅ Model tersimpan di model/skin_cancer.h5")
