import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
recognizer=cv2.face.LBPHFaceRecognizer_create();
recognizer.read("recognizer/trainingData.yml")
id=0
#font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,1,1,0,1)
font=cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        if id==1:
            id="Harshit"
        elif id==2:
            id="Gunjan"
        elif id==3:
            id="kanishk"    
        cv2.putText(img,str(id),(x,y+h),font,2,255,2)
    cv2.imshow("face",img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
    
cam.release()
cv2.destroyAllWindows()