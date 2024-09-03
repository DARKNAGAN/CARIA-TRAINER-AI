import cv2
import numpy as np
import random
import os

# Taille des images
size = 60
# Chemin vers la vidéo et le répertoire de sortie
video = "server-ia/data/videos/autoroute.mp4"
dir_images_genere_sans_panneaux = "server-trainer/images/genere_sans_panneaux"

# Nombre total d'images à générer
nbr_image = 2000

# Création du répertoire de sortie s'il n'existe pas
if not os.path.isdir(dir_images_genere_sans_panneaux):
    os.makedirs(dir_images_genere_sans_panneaux)
    print(f"Répertoire créé : {dir_images_genere_sans_panneaux}")

# Vérification de l'existence de la vidéo
if not os.path.exists(video):
    print(f"Vidéo non présente : {video}")
    quit()

print(f"Vidéo trouvée : {video}")

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    print(f"Erreur lors de l'ouverture de la vidéo : {video}")
    quit()

# Calcul du nombre d'images à générer par frame
nbr_frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
nbr_image_par_frame = int(nbr_image / nbr_frame_total) + 1

print(f"Nombre d'images à générer : {nbr_image}")
print(f"Nombre d'images à générer par frame : {nbr_image_par_frame}")

id = 0

# Variables pour calculer les dimensions globales
dimensions = []
min_w = float('inf')
max_w = 0
min_h = float('inf')
max_h = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin de la vidéo ou erreur de lecture.")
        break

    h, w, c = frame.shape

    # Mettre à jour les dimensions minimales et maximales
    min_w = min(min_w, w)
    max_w = max(max_w, w)
    min_h = min(min_h, h)
    max_h = max(max_h, h)

    # Ajouter les dimensions de la frame
    dimensions.append((w, h, c))

    for _ in range(nbr_image_par_frame):
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        img = frame[y:y + size, x:x + size]
        
        # Sauvegarde de l'image
        img_filename = os.path.join(dir_images_genere_sans_panneaux, f"{id}.png")
        cv2.imwrite(img_filename, img)
        id += 1
        
        if id >= nbr_image:
            print(f"Nombre d'images générées atteint : {nbr_image}")
            break
    
    if id >= nbr_image:
        break

cap.release()

# Affichage du nombre total d'images sauvegardées et de l'emplacement du répertoire
print(f"Nombre total d'images sauvegardées : {id}")
print(f"Emplacement des images sauvegardées : {dir_images_genere_sans_panneaux}")

# Calcul des dimensions globales
average_w = np.mean([w for w, h, c in dimensions])
average_h = np.mean([h for w, h, c in dimensions])

# Affichage des statistiques globales
print(f"Dimensions minimales : {min_w}x{min_h}")
print(f"Dimensions maximales : {max_w}x{max_h}")
print(f"Dimensions moyennes : {average_w:.2f}x{average_h:.2f}")

print("Libération des ressources et fin du traitement.")
