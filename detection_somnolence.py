import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# ceci est utilisé pour obtenir un bip sonore (lorsque la personne ferme les yeux pendant plus de 10 secondes)
mixer.init()
alarm_sound = mixer.Sound('alarm.wav')

# Ces fichies xml sont utilisés pour détecter le visage, l'oeil gauche et l'oeil droite:
face_detection = cv2.CascadeClassifier('haar_cascade_files\haarcascade_frontalface_alt.xml')
left_eye_detection = cv2.CascadeClassifier('haar_cascade_files\haarcascade_lefteye_2splits.xml')
right_eye_detection = cv2.CascadeClassifier('haar_cascade_files\haarcascade_righteye_2splits.xml')

labels =['Close','Open']

# Charger le modèle, que nous avons créé:
model = load_model('models/cnndd.h5')

path = os.getcwd()

# pour capturer chaque image
capture = cv2.VideoCapture(0)

#vérifier si la webcam s'ouvre correctement
if capture.isOpened():
    capture = cv2.VideoCapture(0)
if not capture.isOpened():
    raise IOError("Cannot open webcam")

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


#déclarer des variables
counter = 0
time = 0
thick = 2
right_eye_pred=[99]
left_eye_pred=[99]

while(True):
    ret, frame = capture.read()
    height,width = frame.shape[:2] 

    #convertir l'image capturée en couleur grise :
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # effectuer la détection (cela renverra les coordonnées x, y, la hauteur, la largeur de l'objet des boîtes aux limites)

    faces = face_detection.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = left_eye_detection.detectMultiScale(gray)
    right_eye =   right_eye_detection.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (100,height) , (0,0,0) , thickness=cv2.FILLED )
    cv2.rectangle(frame, (290,height-50) , (540,height) , (0,0,0) , thickness=cv2.FILLED )

#itérer sur les faces et tracer des cadres de délimitation pour chaque face 
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
        
    
#itérer sur l'oeil droit
    for (x,y,w,h) in right_eye:
        
#retirez l'image de l'œil droit du cadre :
        right_one=frame[y:y+h,x:x+w]
        counter += 1
        right_one = cv2.cvtColor(right_one,cv2.COLOR_BGR2GRAY)
        right_one = cv2.resize(right_one,(24,24))
        right_one = right_one/255
        right_one =  right_one.reshape(24,24,-1)
        right_one = np.expand_dims(right_one,axis=0)
        right_eye_pred = np.argmax(model.predict(right_one),axis=-1)
        if(right_eye_pred[0] == 1):
            labels = 'Open' 
        if(right_eye_pred[0]==0):
            labels = 'Closed'
        break

    
#itérer sur l'œil gauche :
    for (x,y,w,h) in left_eye:
        #retirez l'image de l'œil gauche du cadre :
        left_one=frame[y:y+h,x:x+w]
        counter += 1
        left_one = cv2.cvtColor(left_one,cv2.COLOR_BGR2GRAY)  
        left_one = cv2.resize(left_one,(24,24))
        left_one = left_one/255
        left_one = left_one.reshape(24,24,-1)
        left_one = np.expand_dims(left_one,axis=0)
        left_eye_pred =np.argmax(model.predict(left_one),axis=-1)
        if(left_eye_pred[0] == 1):
            labels ='Open'   
        if(left_eye_pred[0] == 0):
            labels ='Closed'
        break

    if(right_eye_pred[0] == 0 and left_eye_pred == 0):
        time += 1
        cv2.putText(frame,"Inactif",(20,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(right_eye_pred[0]==1 or left_eye_pred[0]==1):
    else:
        time -= 1
        cv2.putText(frame,"Actif",(20,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
    
        
    if(time<0):
        time=0   
    cv2.putText(frame,'Attention !!:'+str(time),(300,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)
    if(time>10):
#personne se sent étourdie nous allons alerter :
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            alarm_sound.play()
            
        except:  # isplaying = False
            pass
        if(thick < 16):
            thick = thick+2
        else:
            thick=thick-2
            if(thick<2):
                thick=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thick)
    cv2.imshow('Detection de la Somnolence au Volant_PFE_ENSIAS_MSEA_2022',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()
