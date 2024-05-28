import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils import shuffle
import os
import cv2
import dataset
###GENERE DES IMAGES AVEC PANNEAUX DEPUIS UN MEDIA###
# Tuto25[OpenCV] Lecture des panneaux de vitesse p.1 (houghcircles) 28min
# Taille des images
size = 42
# Chemin vers le répertoire contenant les images de panneaux
dir_images_panneaux = "server-trainer/images/road_sign_speed_trainers/panneaux"
dir_images_genere_panneaux = "server-trainer/images/road_sign_speed_trainers/genere_panneaux"

# Fonction pour lire les images de panneaux à partir du répertoire spécifié
def lire_images_panneaux(dir_images_panneaux, size=None):
    tab_panneau = []
    tab_image_panneau = []

    # Vérifie si le répertoire existe
    if not os.path.exists(dir_images_panneaux):
        quit("Le repertoire d'image n'existe pas: {}".format(dir_images_panneaux))

    # Liste tous les fichiers dans le répertoire
    files = os.listdir(dir_images_panneaux)
    
    # Quitte si le répertoire est vide
    if files is None:
        quit("Le repertoire d'image est vide: {}".format(dir_images_panneaux))

    # Parcours des fichiers dans le répertoire
    for file in sorted(files):
        # Vérifie si le fichier est un fichier PNG
        if file.endswith("png"):
            # Ajoute le nom du fichier (sans l'extension) à tab_panneau
            tab_panneau.append(file.split(".")[0])
            
            # Lit l'image et redimensionne si une taille est spécifiée
            image = cv2.imread(dir_images_panneaux + "/" + file)
            if size is not None:
                image = cv2.resize(image, (size, size), cv2.INTER_LANCZOS4)
            tab_image_panneau.append(image)
            
    return tab_panneau, tab_image_panneau

# Appel de la fonction pour lire les images de panneaux
tab_panneau, tab_image_panneau = lire_images_panneaux(dir_images_panneaux, size)

# Initialisation des tableaux pour les images et les labels
tab_images = np.array([]).reshape(0, size, size, 3)
tab_labels = []

# Génération des données d'entraînement
id = 0
for image in tab_image_panneau:
    lot = []
    for _ in range(100):
        lot.append(dataset.modif_img(image))
    lot = np.array(lot)
    tab_images = np.concatenate([tab_images, lot])
    tab_labels = np.concatenate([tab_labels, np.full(len(lot), id)])
    id += 1

# Conversion des tableaux en type approprié et normalisation des images
tab_panneau = np.array(tab_panneau)
tab_images = np.array(tab_images, dtype=np.float32) / 255
tab_labels = np.array(tab_labels).reshape([-1, 1])

# Mélange des données
tab_images, tab_labels = shuffle(tab_images, tab_labels)

# Boucle pour sauvegarder les images générées dans le répertoire de sortie
for i in range(len(tab_images)):
    # Générer un nom de fichier unique en fonction de l'ID et du nom du panneau
    file_name = "{:d}_{}.png".format(i, tab_panneau[int(tab_labels[i])])
    # Enregistrer l'image dans le répertoire de sortie
    cv2.imwrite(os.path.join(dir_images_genere_panneaux, file_name), tab_images[i] * 255.0)  # Retour à l'échelle 0-255

# Affichage des images avec leur label
for i in range(len(tab_images)):
    cv2.imshow("panneau", tab_images[i])
    print("label", tab_labels[i], "panneau", tab_panneau[int(tab_labels[i])])
    if cv2.waitKey() & 0xFF == ord('q'):
        quit()