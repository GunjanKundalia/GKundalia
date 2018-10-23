import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
#faceDetect=cv2.CascadeClassifier('haarcascade_eye.xml');
cam=cv2.VideoCapture(0);

sampleNo=0;
#identifier
id=input('Enter user id')
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    
    for(x,y,w,h) in faces:
        sampleNo=sampleNo+1
        cv2.imwrite("dataset/User."+str(id)+"."+str(sampleNo)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100);
    cv2.imshow("face",img)
    cv2.waitKey(1);
    if(sampleNo>100):
        break;
    
cam.release()
cv2.destroyAllWindows()