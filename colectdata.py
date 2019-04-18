import cv2
import os
from PIL import Image
import numpy as np
from createfolder import createFolder
def colectdata():
    imge_dit=r'C:\Users\roy\Desktop\Python\opencv2\data'
    y_laibel=[]
    x_train=[]
    counter=0
    font = cv2.FONT_HERSHEY_SIMPLEX
    name=input('enter your name: ')
    if len(name)>0:
        createFolder(r'./data/'+name+'/')
        cap=cv2.VideoCapture(0)
        face_c=cv2.CascadeClassifier(r'C:\Users\roy\Desktop\Python\opencv2\data\haarcascade_frontalface_alt2.xml')
        #taike a photos
        while True:
            ret,fraime=cap.read()
            gray=cv2.cvtColor(fraime,cv2.COLOR_BGR2GRAY)
            face=face_c.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
            for(x,y,w,h) in face:
                print(x,y,w,h)
                roi=gray[y:y+h,x:x+w]
                cv2.rectangle(fraime,(x+w,y+h),(x,y),(255,0,9),2)
            if  cv2.waitKey(1) & 0xFF==ord('p'):
                counter+=1
                cv2.imwrite(r'./data/'+name+'/'+str(counter)+'.jpg',roi)
            cv2.putText(fraime,str(counter),(50,50), font, 1, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow('colectdata',fraime)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        #-------------------------------------------------------------------------------------
        cv2.destroyAllWindows()
