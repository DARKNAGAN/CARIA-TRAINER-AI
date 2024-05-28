import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import os
import time
import dataset

size=42
dir_images_panneaux="server-trainer/images/road_sign_speed_trainers/panneaux"
dir_images_autres_panneaux="server-trainer/images/road_sign_speed_trainers/autres_panneaux"
dir_images_genere_sans_panneaux="server-trainer/images/road_sign_speed_trainers/genere_sans_panneaux"

batch_size=128
nbr_entrainement=1 #20

def panneau_model(nbr_classes):
    model=tf.keras.Sequential()

    model.add(layers.Input(shape=(size, size, 3), dtype='float32'))
    
    model.add(layers.Conv2D(128, 3, strides=1))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(128, 3, strides=1))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    model.add(layers.Conv2D(256, 3, strides=1))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, 3, strides=1))
    model.add(layers.Dropout(0.4))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(nbr_classes, activation='sigmoid'))
    
    return model

def lire_images_panneaux(dir_images_panneaux, size=None):
    tab_panneau=[]
    tab_image_panneau=[]

    if not os.path.exists(dir_images_panneaux):
        quit("Le repertoire d'image n'existe pas: {}".format(dir_images_panneaux))

    files=os.listdir(dir_images_panneaux)
    if files is None:
        quit("Le repertoire d'image est vide: {}".format(dir_images_panneaux))

    for file in sorted(files):
        if file.endswith("png"):
            tab_panneau.append(file.split(".")[0])
            image=cv2.imread(dir_images_panneaux+"/"+file)
            if size is not None:
                image=cv2.resize(image, (size, size), cv2.INTER_LANCZOS4)
            tab_image_panneau.append(image)
            
    return tab_panneau, tab_image_panneau

tab_panneau, tab_image_panneau=lire_images_panneaux(dir_images_panneaux, size)

tab_images=np.array([]).reshape(0, size, size, 3)
tab_labels=np.array([]).reshape(0, len(tab_image_panneau))

id=0
for image in tab_image_panneau:
    lot = []
    for _ in range(120):
        lot.append(dataset.modif_img(image))
    lot = np.array(lot)
    tab_images=np.concatenate((tab_images, lot))
    tab_labels=np.concatenate([tab_labels, np.repeat([np.eye(len(tab_image_panneau))[id]], len(lot), axis=0)])
    id += 1


files=os.listdir(dir_images_autres_panneaux)
if files is None:
    quit("Le repertoire d'image est vide:".format(dir_images_autres_panneaux))

nbr=0
for file in files:
    lot = []
    if file.endswith("png"):
        path=os.path.join(dir_images_autres_panneaux, file)
        image=cv2.resize(cv2.imread(path), (size, size), cv2.INTER_LANCZOS4)
        for _ in range(700):
            lot.append(dataset.modif_img(image))
        lot = np.array(lot)
        tab_images=np.concatenate([tab_images, lot])
        nbr+=len(lot)

tab_labels=np.concatenate([tab_labels, np.repeat([np.full(len(tab_image_panneau), 0)], nbr, axis=0)])

nbr_np=int(len(tab_images)/2)

id=1
nbr=0
tab=[]
for cpt in range(nbr_np):
    file=dir_images_genere_sans_panneaux+"/{:d}.png".format(id)
    if not os.path.isfile(file):
        break
    image=cv2.resize(cv2.imread(file), (size, size))
    tab.append(image)
    id+=1
    nbr+=1

tab_images=np.concatenate([tab_images, tab])
tab_labels=np.concatenate([tab_labels, np.repeat([np.full(len(tab_image_panneau), 0)], nbr, axis=0)])

tab_panneau=np.array(tab_panneau)
tab_images=np.array(tab_images, dtype=np.float32)/255
tab_labels=np.array(tab_labels, dtype=np.float32) #.reshape([-1, 1])

train_images, test_images, train_labels, test_labels=train_test_split(tab_images, tab_labels, test_size=0.10)

train_ds=tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
test_ds=tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

print("train_images", len(train_images))
print("test_images", len(test_images))

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions=model_panneau(images)
    loss=my_loss(labels, predictions)
  gradients=tape.gradient(loss, model_panneau.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model_panneau.trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)

def train(train_ds, nbr_entrainement):
  for entrainement in range(nbr_entrainement):
    start=time.time()
    for images, labels in train_ds:
      train_step(images, labels)
    message='Entrainement {:04d}: loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
    print(message.format(entrainement+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         time.time()-start))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test(test_ds)

def my_loss(labels, preds):
    labels_reshape=tf.reshape(labels, (-1, 1))
    preds_reshape=tf.reshape(preds, (-1, 1))
    result=loss_object(labels_reshape, preds_reshape)
    return result
    
def test(test_ds):
  start=time.time()
  for test_images, test_labels in test_ds:
    predictions=model_panneau(test_images)
    t_loss=my_loss(test_labels, predictions)
    test_loss(t_loss)
    test_accuracy(test_labels, predictions)
  message='   >>> Test: loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
  print(message.format(test_loss.result(),
                       test_accuracy.result()*100,
                       time.time()-start))
  test_loss.reset_states()
  test_accuracy.reset_states()

optimizer=tf.keras.optimizers.Adam()
loss_object=tf.keras.losses.BinaryCrossentropy()
train_loss=tf.keras.metrics.Mean()
train_accuracy=tf.keras.metrics.BinaryAccuracy()
test_loss=tf.keras.metrics.Mean()
test_accuracy=tf.keras.metrics.BinaryAccuracy()
model_panneau=panneau_model(len(tab_panneau))
checkpoint=tf.train.Checkpoint(model_panneau=model_panneau)

print("Entrainement")
train(train_ds, nbr_entrainement)
checkpoint.save(file_prefix="server-ia/data/modeles/road_sign_speed_trainers/modele_panneau")

quit()

for i in range(len(test_images)):
    prediction=model_panneau(np.array([test_images[i]]))
    print("prediction", prediction[0])
    if np.sum(prediction[0])<0.6:
        print("Ce n'est pas un panneau")
    else:
        print("C'est un panneau:", tab_panneau[np.argmax(prediction[0])])
    cv2.imshow("image", test_images[i])
    if cv2.waitKey()&0xFF==ord('q'):
        break
