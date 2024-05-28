import os

# Assurez-vous d'être dans le répertoire Darknet
os.chdir("chemin/vers/votre/darknet")

# Entraînement du modèle YOLOv3 pour les panneaux de signalisation routière
# Utilisation du CPU uniquement

# Chemin vers les données d'entraînement et de validation
train_data = "server-trainer/images/road_sign_trainers/train_speed"
valid_data = "server-trainer/images/road_sign_trainers/test_speed"

# Paramètres d'entraînement
batch_size = 64
subdivisions = 16
num_epochs = 1000

# Commande d'entraînement
train_command = f"./darknet detector train data/obj.data cfg/yolov3_custom_train.cfg yolov3.weights -map -dont_show -gpus 0"

# Boucle d'entraînement
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Entraînement sur les données d'entraînement
    os.system(train_command)

    # Validation sur les données de validation
    os.system(f"./darknet detector map data/obj.data cfg/yolov3_custom_test.cfg backup/yolov3_custom_train_{epoch+1}.weights")

print("Entraînement terminé!")
