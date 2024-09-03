import cv2
import os
import numpy as np
import pickle
from pathlib import Path

# Définir les chemins
IMAGE_DIR = Path(r"W:/CARIA/images/avatars")
LABELS_FILE = Path("server-ia/data/modeles/CV2/labels.pickle")
TRAINER_FILE = Path("server-ia/data/modeles/CV2/trainner.yml")

# Initialiser les variables
current_id = 0
label_ids = {}
x_train = []
y_labels = []

def process_images(image_dir):
    global current_id, label_ids, x_train, y_labels
    print("Début du traitement des images...")
    for root, _, files in os.walk(image_dir):
        if files:
            label = Path(root).name
            for file in files:
                if file.lower().endswith(".jpg"):
                    path = Path(root) / file
                    if label not in label_ids:
                        label_ids[label] = current_id
                        current_id += 1
                    id_ = label_ids[label]
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        x_train.append(image)
                        y_labels.append(id_)
    print(f"Traitement terminé. {len(x_train)} images chargées.")

def save_data():
    print("Sauvegarde des étiquettes...")
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_ids, f)
    print(f"Étiquettes sauvegardées dans {LABELS_FILE}.")

def train_recognizer():
    print("Entraînement du reconnaisseur...")
    x_train_np = np.array(x_train)
    y_labels_np = np.array(y_labels)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(x_train_np, y_labels_np)
    recognizer.save(str(TRAINER_FILE))
    print(f"Modèle entraîné et sauvegardé dans {TRAINER_FILE}.")

def main():
    process_images(IMAGE_DIR)
    save_data()
    train_recognizer()
    print("Traitement complet. Les modèles ont été sauvegardés.")

if __name__ == "__main__":
    main()
