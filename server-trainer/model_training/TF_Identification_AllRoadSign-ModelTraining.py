import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
import time
from tqdm import tqdm

# Fonction pour charger les labels depuis un fichier CSV
def load_labels_from_csv(filepath):
    filenames, labels = [], []
    with open(filepath, 'r') as file:
        next(file)  # Skip header
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 2:
                filenames.append(parts[0])
                labels.append(parts[1])
    return filenames, labels

# Chargement des fichiers CSV des labels
panneaux_filenames, panneaux_labels = load_labels_from_csv('server-trainer/images/vitesse_panneaux_labels.csv')
autres_filenames, autres_labels = load_labels_from_csv('server-trainer/images/autres_panneaux_labels.csv')
sans_filenames, sans_labels = load_labels_from_csv('server-trainer/images/genere_sans_panneaux_labels.csv')

# Création d'un dictionnaire pour les labels
def create_label_dict(filenames, labels):
    label_dict = {i: label for i, label in enumerate(labels)}
    return label_dict

# Dictionnaires de labels
panneaux_labels_dict = create_label_dict(panneaux_filenames, panneaux_labels)
autres_labels_dict = create_label_dict(autres_filenames, autres_labels)
sans_labels_dict = {0: "Sans panneau"}

# Configuration
size = 60
dir_images_panneaux = "server-trainer/images/vitesse_panneaux"
dir_images_autres_panneaux = "server-trainer/images/autres_panneaux"
dir_images_genere_sans_panneaux = "server-trainer/images/genere_sans_panneaux"
batch_size = 128
nbr_entrainement = 1  # Nombre d'époques d'entraînement

print(f"Configuration : taille des images = {size}, taille du batch = {batch_size}, nombre d'époques = {nbr_entrainement}")

# Définition du modèle
def panneau_model(nbr_classes):
    print(f"Création du modèle avec {nbr_classes} classes.")
    model = tf.keras.Sequential([
        layers.Input(shape=(size, size, 3)),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(nbr_classes, activation='sigmoid')
    ])
    print("Modèle créé.")
    return model

# Chargement des images depuis un répertoire
def load_images_from_directory(directory, size):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Le répertoire d'images n'existe pas : {directory}")
    
    files = [f for f in sorted(os.listdir(directory)) if f.endswith(".png")]
    if not files:
        raise FileNotFoundError(f"Le répertoire d'images est vide : {directory}")
    
    images = []
    for file in files:
        path = os.path.join(directory, file)
        image = cv2.imread(path)
        image = cv2.resize(image, (size, size), cv2.INTER_LANCZOS4)
        images.append(image)
    
    return images

# Préparation des données
def prepare_data(panneaux_images, autres_images, sans_panneaux_images, panneau_labels_dict):
    print("Préparation des données pour les images de panneaux...")
    tab_images = np.array([])
    tab_labels = np.array([])

    # Images de panneaux
    for i, image in enumerate(panneaux_images):
        lot = np.array([image for _ in range(120)])
        if tab_images.size == 0:
            tab_images = lot
        else:
            tab_images = np.concatenate((tab_images, lot), axis=0)
        
        labels = np.eye(len(panneaux_labels_dict))[i]
        labels = np.repeat([labels], len(lot), axis=0)
        if tab_labels.size == 0:
            tab_labels = labels
        else:
            tab_labels = np.concatenate([tab_labels, labels], axis=0)

    print(f"Nombre d'images après ajout des panneaux : {len(tab_images)}")
    print(f"Nombre de labels après ajout des panneaux : {len(tab_labels)}")

    # Images des autres panneaux
    print("Traitement des autres panneaux...")
    for image in autres_images:
        lot = np.array([image for _ in range(700)])
        if tab_images.size == 0:
            tab_images = lot
        else:
            tab_images = np.concatenate([tab_images, lot], axis=0)
        
        labels = np.zeros((len(lot), len(panneaux_labels_dict)))
        if tab_labels.size == 0:
            tab_labels = labels
        else:
            tab_labels = np.concatenate([tab_labels, labels], axis=0)

    print(f"Nombre d'images après ajout des autres panneaux : {len(tab_images)}")
    print(f"Nombre de labels après ajout des autres panneaux : {len(tab_labels)}")

    # Images générées sans panneaux
    print("Traitement des images générées sans panneaux...")
    sans_panneaux_images = [cv2.resize(img, (size, size)) for img in sans_panneaux_images]
    tab_images = np.concatenate([tab_images, np.array(sans_panneaux_images)], axis=0)
    tab_labels = np.concatenate([tab_labels, np.zeros((len(sans_panneaux_images), len(panneaux_labels_dict)))], axis=0)

    print(f"Nombre d'images après ajout des images sans panneaux : {len(tab_images)}")
    print(f"Nombre de labels après ajout des images sans panneaux : {len(tab_labels)}")

    # Normalisation
    tab_images = tab_images.astype(np.float32) / 255
    tab_labels = tab_labels.astype(np.float32)
    
    return tab_images, tab_labels

# Chargement des images
panneaux_images = load_images_from_directory(dir_images_panneaux, size)
autres_images = load_images_from_directory(dir_images_autres_panneaux, size)
sans_panneaux_images = load_images_from_directory(dir_images_genere_sans_panneaux, size)

# Préparation des données
tab_images, tab_labels = prepare_data(panneaux_images, autres_images, sans_panneaux_images, panneaux_labels_dict)

# Division du jeu de données
train_images, test_images, train_labels, test_labels = train_test_split(tab_images, tab_labels, test_size=0.10, random_state=42)

print(f"Nombre d'images d'entraînement : {len(train_images)}, Nombre d'images de test : {len(test_images)}")

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

# Compilation du modèle
model_panneau = panneau_model(len(panneaux_labels_dict))
model_panneau.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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

model_panneau.save("server-ia/data/modeles/tensorflow/tf_modele_AllRoadSign.keras")
print(f"Modèle sauvegardé à : {os.path.abspath(model_panneau)}")

# Évaluation des prédictions
print("Évaluation des prédictions sur les images de test.")
for i in range(len(test_images)):
    prediction = model_panneau.predict(np.array([test_images[i]]))
    predicted_index = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    # Affichage du label correspondant
    if confidence < 0.6:
        print("Ce n'est pas un panneau")
    else:
        print(f"C'est un panneau : {panneaux_labels_dict.get(predicted_index, 'Inconnu')}")

    # Affichage de l'image
    cv2.imshow("image", test_images[i])
    if cv2.waitKey() & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
