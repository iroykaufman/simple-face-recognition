import os
import cv2
import numpy as np
def trainprodict():
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        faces = []
        labels = []
        countre=0
        subjects=[""]
        cap=cv2.VideoCapture(0)
        face_c=cv2.CascadeClassifier(r'C:\Users\roy\Desktop\Python\opencv2\data\haarcascade_frontalface_alt2.xml')
        path = '.\data'
        files = os.listdir(path)
        for namef in files:
                countre+=1
                subjects.append(namef)
                #subjects.append(namef)
                #labels.append(countre)
                #print(labels)
                path=r".\data" + '\\' +namef
                files2 = os.listdir(path)
                for nameim in files2:
                        patth=path+'\\'+nameim 
                        image=cv2.imread(patth,cv2.IMREAD_GRAYSCALE)
                        faces.append(image)
                        #namef=int(namef)
                        labels.append(countre)
                        

        print("Lets gather some data")
        print("Total data to train: ", len(faces))
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(faces, np.array(labels))
        while True:
                ret,fraime=cap.read()
                gray=cv2.cvtColor(fraime,cv2.COLOR_BGR2GRAY)
                face=face_c.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
                for(x,y,w,h) in face:
                        roi=gray[y:y+h,x:x+w]
                        label, confidence = face_recognizer.predict(roi)
                        if (confidence < 100):
                                label = subjects[label]
                                confidence = "  {0}%".format((round(confidence)))
                                
                        else:
                                label = subjects[label]
                                confidence = "  {0}%".format(abs(round(100 - confidence)))
                        
                        
                        cv2.rectangle(fraime,(x+w,y+h),(x,y),(255,0,9),2)
                        cv2.putText(fraime,label+"-"+confidence,(x,y), font, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow('fraime',fraime)
                if cv2.waitKey(1) & 0xFF==ord('q'):
                        
                        break



