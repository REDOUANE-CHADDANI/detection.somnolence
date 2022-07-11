# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 19:16:16 2022

@author: REDOUANE-CH
"""

# importation des bibliothèques requises :
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
import pickle
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization

# définir le chemin d'accès à notre ensemble de données sur les yeux :
Directory = r'C:\Users\REDOUANE-CH\OneDrive\Bureau\detection_de_la_somnolence_PFE\Base_de_donnees\train'

# spécifiez deux catégories sur lesquelles nous voulons entraîner nos données :
CATEGORIES = ['Closed' , 'Open']

#définir la taille de l'image :
img_size = 24
data = []


#itérer sur chaque image et obtenir l'image sous forme de tableau,
for category in CATEGORIES:
    folder = os.path.join(Directory,category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        img_arr = cv2.resize(img_arr,(img_size, img_size),1)
        data.append([img_arr , label])

# voir la longueur des données :
len(data)
# nous mélangeons les données pour obtenir des images aléatoires d'yeux ouverts et d'yeux fermés :
random.shuffle(data)

# divisant les fonctionnalités et l'étiquette pour la formation du modèle :
X = []
Y = []

for features,label in data:
    X.append(features)
    Y.append(label)

#convertissez-les en tableau :
X = np.array(X)
Y = np.array(Y)


# enregistrer les données dans le système :
pickle.dump(X , open('X.pkl' , 'wb'))
pickle.dump(Y , open('Y.pkl' , 'wb'))


# normaliser le tableau d'images :
X = X/255

# remodeler le tableau X en (24,24,1)
img_rows,img_cols = 24,24
X = X.reshape(X.shape[0],img_rows,img_cols,1)
X.shape

# modèle de création :
model = Sequential()

model.add(Conv2D(64 , (3,3) , activation = 'relu' , input_shape= X.shape[1:]))
model.add(MaxPooling2D((1,1)))

model.add(Conv2D(64 , (3,3) , activation = 'relu'))
model.add(MaxPooling2D((1,1)))

model.add(Conv2D(64 , (3,3) , activation = 'relu'))
model.add(MaxPooling2D((1,1)))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))

# modèle de compilation que nous avons créé
model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])


# ajuster X , Y au modèle pour voir la précision du modèle :
model.fit(X, Y, epochs = 5 , validation_split = 0.1 , batch_size = 32)


# enregistrer le modèle et l'architecture dans un seul fichier
model.save("custmodel.h5")



