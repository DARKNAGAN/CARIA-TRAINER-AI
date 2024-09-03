import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np

# Définir les chemins des données d'entraînement et de validation
train_data_dir = "server-trainer/images/road_sign_trainers/train_full"
valid_data_dir = "server-trainer/images/road_sign_trainers/test_full"

# Initialiser les hyperparamètres
INIT_LR = 1e-4
EPOCHS = 1
BS = 32

# Charger et prétraiter les images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=BS,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(224, 224),
    batch_size=BS,
    class_mode='categorical'
)

# Générer le fichier class_names.txt
class_names_file = os.path.join("server-ia/data/modeles/MobileNetV2", "class_names.txt")
with open(class_names_file, "w") as f:
    # Utiliser le générateur class_indices pour récupérer les noms de classe et les écrire dans le fichier
    for class_name, class_index in train_generator.class_indices.items():
        f.write(f"{class_name}\n")

# Charger le modèle pré-entraîné MobileNetV2 sans la couche supérieure
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

# Construire le modèle de tête qui sera placé au-dessus du modèle de base
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(len(train_generator.class_indices), activation="softmax")(head_model)

# Combiner le modèle de base avec le modèle de tête
model = Model(inputs=base_model.input, outputs=head_model)

# Geler les couches du modèle de base
for layer in base_model.layers:
    layer.trainable = False

# Compiler le modèle
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Entraîner le modèle
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    epochs=EPOCHS
)

# Sauvegarder le modèle
model.save("server-ia/data/modeles/MobileNetV2/modele_signaux_routiers.keras")

# Évaluer le modèle
print("[INFO] Évaluation du modèle...")
predictions = model.predict(valid_generator, steps=len(valid_generator), verbose=1)
predictions = np.argmax(predictions, axis=1)
print(classification_report(valid_generator.classes, predictions, target_names=valid_generator.class_indices.keys()))
