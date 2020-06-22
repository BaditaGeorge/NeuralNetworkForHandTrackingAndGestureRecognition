import imageProcessor
import numpy as np
import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.utils import to_categorical
import pandas as pd
import os
import cv2
from PIL import Image
from resizeimage import resizeimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class HandLocation:
    def __init__(self):
        self.imgPths = []
        self.imgArr = []
        self.data = []
        self.labels = []
        self.train_data = []
        self.test_data = []
        self.train_labels = []
        self.test_labels = []

    #we don't use that here

    def toScale(self,maxN,v):
        return v/maxN

    def retrievePaths(self,path):
        for r,d,f in os.walk(path):
            for fl in f:
                path = os.path.join(r,fl)
                if path.endswith('png') or path.endswith('jpg') or path.endswith('bmp'):
                    self.imgPths.append(path)

    def plotImage(self,index):
        img = cv2.imread(self.imgPths[index])
        img_cvt = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        plt.imshow(img_cvt)
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.title("Image " + self.imgPths[index])
        plt.show()

    def getLabel(self,pth):
        section = pth.split('\\')[2]
        label = section.split('_')[0][1]
        return label

    def prepareData(self):
        fl = open('labels','r')
        for imgPth in self.imgPths:
            img = cv2.imread(imgPth)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(320,120))
            self.data.append([img])
            arr = fl.readline().split(' ')
            tpl = []
            tpl.append(self.toScale(1280,int(float(arr[0]))))
            tpl.append(self.toScale(720,int(float(arr[1]))))
            tpl.append(self.toScale(720,int(float(arr[2][:len(arr[2])-1]))))
            self.labels.append(tpl)
        self.data = np.array(self.data,dtype="uint8")
        self.data = self.data.reshape(len(self.imgPths),120,320,1)
        self.labels = np.asarray(self.labels)
        # print(self.labels)
        # print(self.data)

    def slideThrough(self,index):
        while 1:
            cv2.imshow('img',self.data[index])
            k_pr = cv2.waitKey(1)
            if k_pr == 27:
                break
    
    def splitSets(self,prc):
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.data,self.labels,test_size=prc,random_state=42)

    def setTestData(self,pth):
        self.retrievePaths(pth)
        self.prepareData(False)
        self.test_data = self.data
    
    def train(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120,320,1))) 
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu')) 
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        for layer in model.layers:
            print(layer.output_shape)
        # model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        # model.fit(self.data,self.labels,epochs=5,batch_size=64,verbose=2)
        # model.save('handDetector.h5')

    def loadModel(self,pth):
        print(self.test_data[0])
        model = Sequential()
        model = keras.models.load_model(pth)
        print(np.argmax(model.predict(np.asarray(self.test_data))))



a = HandLocation()
a.retrievePaths('definitive')
a.prepareData()
print(len(a.data))
print(len(a.data[0]))
print(len(a.data[0][0]))
print(len(a.data[0][0][0]))
a.train()
# print(a.labels)
# a.train()
# a.slideThrough(2)
# a.splitSets(0.3)
# a.setTestData('imos2')
# a.train()
# a.loadModel('rnModel.h5')
# a.plotImage(1)