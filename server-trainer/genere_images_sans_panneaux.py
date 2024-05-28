import cv2
import numpy as np
import random
import os
###GENERE DES IMAGES SANS PANNEAUX DEPUIS UN MEDIA###
#YT:Tuto25[Tensorflow2] Lecture des panneaux de vitesse p.2 - 4min30

size=42
video="server-trainer/videos/autoroute.mp4"
dir_images_genere_sans_panneaux="server-trainer/images/road_sign_speed_trainers/genere_sans_panneaux"

if not os.path.isdir(dir_images_genere_sans_panneaux):
    os.mkdir(dir_images_genere_sans_panneaux)

if not os.path.exists(video):
    print("Vidéo non présente:", video)
    quit()
    
cap=cv2.VideoCapture(video)

id=0
nbr_image=1500

nbr_image_par_frame=int(1500/cap.get(cv2.CAP_PROP_FRAME_COUNT))+1

while True:
    ret, frame=cap.read()
    if ret is False:
        quit()
    h, w, c=frame.shape

    for cpt in range(nbr_image_par_frame):
        x=random.randint(0, w-size)
        y=random.randint(0, h-size)
        img=frame[y:y+size, x:x+size]        
        cv2.imwrite(dir_images_genere_sans_panneaux+"/{:d}.png".format(id), img)
        id+=1
        if id==nbr_image:
            quit()