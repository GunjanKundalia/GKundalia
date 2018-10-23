#to find path of images
import os
import cv2
import numpy as np
from PIL import Image

#create a recognizer

recognizer=cv2.face.LBPHFaceRecognizer_create();
path='dataset'

#create a method to get images
def getImagesWithId(path):
    imagePath=[os.path.join(path,f) for f in os.listdir(path)]
    # print (imagePath)
    faces=[]
    IDs=[]
    for imgPath in imagePath:
        #convert image in numpy array
        faceImg=Image.open(imgPath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        #for having ID we split the path
        ID=int(os.path.split(imgPath)[-1].split('.')[1])
        faces.append(faceNp)
       # print(ID)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(20)
    return np.array(IDs),faces
   
    
Ids,faces=getImagesWithId(path)   
recognizer.train(faces,Ids)
recognizer.save('recognizer/trainingData.yml')

cv2.destroyAllWindows()
