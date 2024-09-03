import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import os
import time
from tqdm import tqdm

# Configuration
size = 60
csv_file = "server-trainer/images/genere_vitesse_panneaux_labels.csv"
image_dir = "server-trainer/images/genere_vitesse_panneaux/"
batch_size = 128
nbr_entrainement = 10  # Nombre d'époques d'entraînement
output_csv = "server-ia/data/modeles/tensorflow/train_labels.csv"

print(f"Configuration : taille des images = {size}, taille du batch = {batch_size}, nombre d'époques = {nbr_entrainement}")

# Définition du modèle
def panneau_model(nbr_classes):
    print(f"Création du modèle avec {nbr_classes} classes.")
    model = tf.keras.Sequential([
        layers.Input(shape=(size, size, 3)),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=2),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=2),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(nbr_classes, activation='softmax')
    ])
    print("Modèle créé.")
    return model

# Chargement des images et des labels depuis le CSV
def load_data_from_csv(csv_file, image_dir, size):
    df = pd.read_csv(csv_file)
    
    # Afficher les noms des colonnes pour débogage
    print("Colonnes dans le CSV :", df.columns)
    
    # Créer un dictionnaire pour mapper les labels alphanumériques à des indices
    labels_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
    reverse_labels_mapping = {idx: label for label, idx in labels_mapping.items()}  # Inverse du mapping pour affichage
    print(f"Mapping des labels : {labels_mapping}")

    images = []
    labels = []
    
    for _, row in df.iterrows():
        # Assurer que les colonnes existent
        if 'filename' not in row or 'label' not in row:
            raise ValueError("Les colonnes 'filename' ou 'label' sont manquantes dans le CSV")
        
        filename = row['filename']
        label = row['label']
        image_path = os.path.join(image_dir, filename)
        
        # Charger et redimensionner l'image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erreur de chargement de l'image : {image_path}")
            continue
        image = cv2.resize(image, (size, size), cv2.INTER_LANCZOS4)
        images.append(image)
        labels.append(labels_mapping[label])
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    # Normalisation
    images /= 255.0
    
    return images, labels, len(labels_mapping), reverse_labels_mapping

# Chargement des données
tab_images, tab_labels, num_classes, reverse_labels_mapping = load_data_from_csv(csv_file, image_dir, size)

# Division du jeu de données
train_images, test_images, train_labels, test_labels = train_test_split(tab_images, tab_labels, test_size=0.10, random_state=42)

print(f"Nombre d'images d'entraînement : {len(train_images)}, Nombre d'images de test : {len(test_images)}")

# Sauvegarde des labels d'entraînement dans un CSV
train_filenames = [f'train_img_{i}.jpg' for i in range(len(train_images))]
train_labels_str = [reverse_labels_mapping[label] for label in train_labels]

df_train = pd.DataFrame({
    'filename': train_filenames,
    'label': train_labels_str
})

# Sauvegarder le CSV
df_train.to_csv(output_csv, index=False)
print(f"Fichier CSV des labels d'entraînement sauvegardé à l'emplacement : {output_csv}")

# Création des datasets TensorFlow
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

# Compilation du modèle
model_panneau = panneau_model(num_classes)
model_panneau.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Modèle compilé.")

# Entraînement du modèle avec barre de progression pour chaque époque
def train(model, train_ds, test_ds, nbr_entrainement):
    for epoch in range(nbr_entrainement):
        print(f"Début de l'entraînement {epoch + 1}...")
        start = time.time()
        
        # Nombre de batches dans une époque
        num_batches = len(train_ds)
        
        # Initialiser tqdm pour les batches
        with tqdm(total=num_batches, desc=f"Époque {epoch + 1}", unit='batch') as pbar:
            for batch_images, batch_labels in train_ds:
                history = model.train_on_batch(batch_images, batch_labels)
                pbar.update(1)  # Mise à jour de la barre de progression
        
        # Afficher les résultats après l'époque
        train_loss, train_accuracy = history
        print(f'Entraînement {epoch + 1:04d} : perte : {train_loss:6.4f}, précision : {train_accuracy * 100:7.4f}%, temps : {time.time() - start:7.4f}')
        test(model, test_ds)

def test(model, test_ds):
    print("Début du test...")
    start = time.time()
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f'   >>> Test : perte : {test_loss:6.4f}, précision : {test_accuracy * 100:7.4f}%, temps : {time.time() - start:7.4f}')

print("Début de l'entraînement du modèle.")
train(model_panneau, train_ds, test_ds, nbr_entrainement)

# Chemin de sauvegarde du modèle
model_save_path = "server-ia/data/modeles/tensorflow/tf_modele_speed_panneau.keras"
model_panneau.save(model_save_path)
print(f"Modèle sauvegardé à l'emplacement : {model_save_path}")

# Évaluation des prédictions
print("Évaluation des prédictions sur les images de test.")
for i in range(len(test_images)):
    prediction = model_panneau.predict(np.array([test_images[i]]))
    predicted_label_index = np.argmax(prediction[0])
    predicted_label = reverse_labels_mapping[predicted_label_index]
    actual_label = reverse_labels_mapping[test_labels[i]]
    print(f"Image {i}: Prédiction = {predicted_label}, Label réel = {actual_label}")
    cv2.imshow("image", test_images[i])
    if cv2.waitKey() & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
