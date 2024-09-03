import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils import shuffle
import os
import cv2
import csv
import dataset_params_edition as dataset

# Taille des images
size = 60
# Nombre total d'images à générer
nombre_images_a_generer = 2000
# Chemin vers le répertoire contenant les images de panneaux
dir_images_panneaux = "server-trainer/images/autres_panneaux"
dir_images_genere_panneaux = "server-trainer/images/genere_autres_panneaux"
csv_file_path = "server-trainer/images/genere_autres_panneaux_labels.csv"

# Fonction pour lire les images de panneaux à partir du répertoire spécifié
def lire_images_panneaux(dir_images_panneaux, size=None):
    print(f"Lecture des images depuis le répertoire : {dir_images_panneaux}")
    tab_panneau = []
    tab_image_panneau = []

    if not os.path.exists(dir_images_panneaux):
        quit(f"Le répertoire d'image n'existe pas: {dir_images_panneaux}")

    files = os.listdir(dir_images_panneaux)
    
    if files is None or len(files) == 0:
        quit(f"Le répertoire d'image est vide: {dir_images_panneaux}")

    for file in sorted(files):
        if file.endswith("png"):
            tab_panneau.append(file.split(".")[0])
            image = cv2.imread(os.path.join(dir_images_panneaux, file))
            if size is not None:
                image = cv2.resize(image, (size, size), cv2.INTER_LANCZOS4)
            tab_image_panneau.append(image)
            
    return tab_panneau, tab_image_panneau

# Supprimer les images existantes et le fichier CSV
if os.path.exists(dir_images_genere_panneaux):
    for file in os.listdir(dir_images_genere_panneaux):
        file_path = os.path.join(dir_images_genere_panneaux, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

if os.path.exists(csv_file_path):
    os.remove(csv_file_path)

tab_panneau, tab_image_panneau = lire_images_panneaux(dir_images_panneaux, size)
print(f"Nombre d'images lues : {len(tab_image_panneau)}")

tab_images = np.array([]).reshape(0, size, size, 3)
tab_labels = []

images_par_panneau = nombre_images_a_generer // len(tab_image_panneau)
print(f"Nombre d'images à générer par panneau : {images_par_panneau}")

print("Génération des images modifiées...")
csv_data = []  # Liste pour stocker les données du CSV
for id, image in enumerate(tab_image_panneau):
    lot = []
    for i in range(images_par_panneau):
        lot.append(dataset.modif_img(image))
    lot = np.array(lot)
    tab_images = np.concatenate([tab_images, lot])
    tab_labels = np.concatenate([tab_labels, np.full(len(lot), id)])

print(f"Nombre total d'images générées : {len(tab_images)}")

tab_panneau = np.array(tab_panneau)
tab_images = np.array(tab_images, dtype=np.float32) / 255
tab_labels = np.array(tab_labels).reshape([-1, 1])

tab_images, tab_labels = shuffle(tab_images, tab_labels)
print("Données mélangées.")

print(f"Sauvegarde des images générées dans : {dir_images_genere_panneaux}")
if not os.path.exists(dir_images_genere_panneaux):
    os.makedirs(dir_images_genere_panneaux)

for i in range(len(tab_images)):
    # Générer un nom de fichier unique
    file_name = "{}_{}.png".format(i, tab_panneau[int(tab_labels[i])])
    # Enregistrer l'image dans le répertoire de sortie
    cv2.imwrite(os.path.join(dir_images_genere_panneaux, file_name), tab_images[i] * 255.0)
    
    # Ajouter les données au CSV
    label = "OP" + str(i)
    csv_data.append([file_name, label])

print("Toutes les images ont été sauvegardées.")
print(f"Nombre total d'images sauvegardées : {len(tab_images)}")

# Écrire les données dans le fichier CSV
print(f"Écriture des labels dans le fichier CSV : {csv_file_path}")
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "label"])  # Écrire l'en-tête
    writer.writerows(csv_data)  # Écrire les données

print("Fichier CSV généré.")

print("Affichage des images générées...")
for i in range(len(tab_images)):
    cv2.imshow("panneau", tab_images[i])
    print(f"label: {tab_labels[i][0]}, panneau: {tab_panneau[int(tab_labels[i])]}")  
    key = cv2.waitKey(0) & 0xFF  
    if key == ord('q'):
        print("Sortie demandée par l'utilisateur.")
        break

cv2.destroyAllWindows()
print("Terminé.")
