import numpy as np
import os
import cv2
from PIL import Image
import pickle, sqlite3


#Xml dosyalarımızı projemize dahil ettik
face_cascade = cv2.CascadeClassifier('Classifiers/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Classifiers/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/training_data.yml")


#Sql serverimızda olan verilerle kıyaslama yaptık
def getProfile(Id):
    conn=sqlite3.connect("DataBase.db")
    query="SELECT * FROM Kullanicilar WHERE ID="+str(Id)
    cursor=conn.execute(query)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

#Video kayıt başlattık
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
 
#Gürüntüde bulunan yüzleri tespit edip dikdörtgen içine aldırdık.
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]


        nbr_predicted, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 70:
            profile=getProfile(nbr_predicted)


   #Databasemizde bulunan kayıtlarla kıyasladık kayıtta yoksa bilinmiyor yazdırdık bilgilerine  
            if profile != None:
                cv2.putText(img, "Ad: "+str(profile[1]), (x, y+h+30), font, 0.4, (0, 0, 255), 1);
                cv2.putText(img, "Yas: " + str(profile[2]), (x, y + h + 50), font, 0.4, (0, 0, 255), 1);
                cv2.putText(img, "Cinsiyet: " + str(profile[3]), (x, y + h + 70), font, 0.4, (0, 0, 255), 1);
                cv2.putText(img, "Sabika Kaydi: " +str(profile[4]), (x, y + h + 90), font, 0.4, (0, 0, 255), 1);
        else:
            cv2.putText(img, "Ad: Bilinmiyor", (x, y + h + 30), font, 0.4, (0, 0, 255), 1);
            cv2.putText(img, "Yas: Bilinmiyor", (x, y + h + 50), font, 0.4, (0, 0, 255), 1);
            cv2.putText(img, "Cinsiyet: Bilinmiyor", (x, y + h + 70), font, 0.4, (0, 0, 255), 1);
            cv2.putText(img, "Sabika Kaydi: Bilinmiyor", (x, y + h + 90), font, 0.4, (0, 0, 255), 1);

    cv2.imshow('Yuz Tanima', img)
    if(cv2.waitKey(1) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()

