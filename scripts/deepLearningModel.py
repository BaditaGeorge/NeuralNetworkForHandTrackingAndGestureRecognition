import imageProcessor
import numpy as np
import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import LocallyConnected1D
from keras.layers import LocallyConnected2D
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.optimizers import Adam
import pandas as pd
import os
import cv2
from PIL import Image
from resizeimage import resizeimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import copy

class NeuralNetwork:
    def __init__(self):
        self.imgPths = []
        self.imgArr = []
        self.data = []
        self.labels = []
        self.train_data = []
        self.test_data = []
        self.train_labels = []
        self.test_labels = []

    def retrievePaths(self,path):
        for r,d,f in os.walk(path):
            for fl in f:
                path = os.path.join(r,fl)
                if path.endswith('png') or path.endswith('jpg') or path.endswith('bmp'):
                    self.imgPths.append(path)

    def anotherRetrievePaths(self,path):
        for i in range(0,10812):
            if os.path.exists(os.path.join(path,'img'+str(i)+'.jpg')):
                self.imgPths.append(os.path.join(path,'img'+str(i)+'.jpg'))

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

    def prepareData(self,labelOn):
        # cnt = 0
        fl = open('labels5','r')
        for imgPth in self.imgPths:
            img = cv2.imread(imgPth)
            # print(img)
            # print(img[0,0])
            # print(img[0])
            # print(img[0][0])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(180,120))
            self.data.append(img)
            arr = fl.readline().split(' ')
            tpl = []
            r = int(float(arr[2][:len(arr[2])-1]))
            lX = int(float(arr[0])) - r
            lY = int(float(arr[1])) - r
            rX = int(float(arr[0])) + r
            rY = int(float(arr[1])) + r
            if lX < 0:
                lX = 0
            if lY < 0:
                lY = 0
            if rX >= 1280:
                rX = 1280
            if rY >= 720:
                rY = 720
            tpl.append(self.toScale(1280,lX))
            tpl.append(self.toScale(720,lY))
            tpl.append(self.toScale(1280,rX))
            tpl.append(self.toScale(720,rY))
            # if (rX - lX) > 0 and (rY - lY) > 0:
            #     while 1:
            #         imgToShow = img[self.fromScale(120,tpl[1]):self.fromScale(120,tpl[3]),self.fromScale(180,tpl[0]):self.fromScale(180,tpl[2])]
            #         cv2.imshow('img',imgToShow)
            #         k_p = cv2.waitKey(1)
            #         if k_p == 27:
            #             break
            self.labels.append(tpl)
            # if labelOn == True:
            #     if cnt <= 194:
            #         self.labels.append(0)
            #     elif cnt > 194 and cnt <= 334:
            #         self.labels.append(1)
            #     elif cnt > 334 and cnt <= 507:
            #         self.labels.append(2)
            #     elif cnt > 507 and cnt <= 687:
            #         self.labels.append(3)
            #     else:
            #         self.labels.append(4)
            # cnt += 1
        self.data = np.array(self.data,dtype="uint8")
        self.data = self.data.reshape(len(self.imgPths),120,180,1)
        self.data = self.data/255
        self.labels = np.asarray(self.labels)

    def fromScale(self,mX,v):
        return int(mX*v)

    def splitSets(self,prc):
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.data,self.labels,test_size=prc,random_state=42)

    def setTestData(self,pth):
        self.retrievePaths(pth)
        self.prepareData(False)
        self.test_data = self.data
    
    def train(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 320, 1))) 
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu')) 
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        for l in model.layers:
            print(l.output_shape)
        return
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.fit(self.train_data,self.train_labels,epochs=5,batch_size=64,verbose=2,validation_data=(self.test_data,self.test_labels))
        model.save('rnModel2.h5')

    def loadModel(self,pth):
        # print(np.asarray([[self.test_data[0][0]]]))
        # print(len(self.test_data))
        # print(len(self.test_data[0]))
        # print(len(self.test_data[1]))
        # print(len(self.test_data[0][0]))
        # print(self.test_data[0])
        model = Sequential()
        model = keras.models.load_model(pth)
        print('Model loaded')
        # print(self.test_data)
        x = model.predict(np.asarray(self.test_data))
        for i in range(len(x)):
            print(np.argmax(x[i]),self.test_labels[i])
        # print(np.argmax(model.predict(np.asarray(self.test_data))))
        # test_loss,test_acc = model.evaluate(self.test_data,self.test_labels)
        # print('Test accuracy: {:2.2f}%'.format(test_acc*100))

    def toScale(self,maxN,v):
        return v/maxN

    def shuffleData(self):
        index = 0
        while index < 9100:
            x = random.randint(0,10792)
            y = random.randint(0,10792)
            while x == y:
                x = random.randint(0,10792)
                y = random.randint(0,10792)
            tmp = copy.deepcopy(self.data[x])
            self.data[x] = copy.deepcopy(self.data[y])
            self.data[y] = tmp
            # self.data[x],self.data[y] = self.data[y],self.data[x]
            tmp = copy.deepcopy(self.labels[x])
            self.labels[x] = copy.deepcopy(self.labels[y])
            self.labels[y] = tmp
            index += 1

    def newRnModelTrial(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(120, 180, 1)))
        model.add(MaxPooling2D((2, 2)))
        # model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        # model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        # model.add(Dense(64,activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='relu'))
        sgd = Adam(lr=0.001)
        # sgd = SGD(lr=0.001,momentum=0.3,nesterov=True)
        model.compile(optimizer=sgd,loss='mse',metrics=['mse'])
        model.fit(self.train_data,self.train_labels,nb_epoch=10,batch_size=64,validation_data=(self.test_data,self.test_labels),verbose=2)
        model.save('handDetector25.h5')

    def retrainModel(self,pth):
        model = Sequential()
        model = keras.models.load_model(pth)
        sgd = Adam(lr=0.001)
        model.compile(optimizer=sgd,loss='mse',metrics=['mse'])
        model.fit(self.train_data,self.train_labels,nb_epoch=10,batch_size=64,validation_data=(self.test_data,self.test_labels),verbose=2)
        model.save('handDetector15.h5')


    def rnModelForTrial(self):
        model = Sequential()
        model.add(Dense(128,input_shape=(120,180,1),activation='relu'))
        # model.add(Dense(200,activation='relu'))
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(32,activation='relu'))
        model.add(Flatten())
        model.add(Dense(3))
        # model.add(LocallyConnected1D(16,3)) 
        for l in model.layers:
            print(l.output_shape)
        # model.compile('adadelta','mse',metrics=['accuracy'])
        # model.fit(self.train_data, self.train_labels, nb_epoch=10, validation_data=(self.test_data,self.test_labels), verbose=2)
        # model.save('handDetector5.h5')
        # x = model.predict(self.test_data[:5])
        # for i in len(x):
        #     print(x[i])
        #     print(self.test_labels[i])
        # for i in model.layers:
        #     print(i.output_shape)

    def trainABitMore(self,pth,secPth):
        img = cv2.imread(secPth)
        originalImg = np.copy(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(180,120))
        dt = []
        dt.append(img)
        dt = np.array(dt,dtype="uint8")
        dt = dt.reshape(len(dt),120,180,1)
        dt = dt/255
        print(dt)
        model = Sequential()
        model = keras.models.load_model(pth)
        x = model.predict(dt)
        print(x)
        while True:
            xl = self.fromScale(1280,x[0][0])
            yl = self.fromScale(720,x[0][1])
            xr = self.fromScale(1280,x[0][2])
            yr = self.fromScale(720,x[0][3])
            while xl < 0:
                xl += 1
            while yl < 0:
                yl += 1
            while xr > 1280:
                xr -= 1
            while yr > 720:
                yr -= 1
            img2 = originalImg[yl:yr,xl:xr]
            cv2.imshow('img',img2)
            k_p = cv2.waitKey(1)
            if k_p == 27:
                break
    

    def trainAndTest(self,modelPth,pth):
        i = 6000
        model = Sequential()
        model = keras.models.load_model(modelPth)
        while i < 7623:
            while not os.path.exists(os.path.join(pth,'img'+str(i)+'.jpg')):
                i += 1
            img = cv2.imread(os.path.join(pth,'img'+str(i)+'.jpg'))
            originalImg = np.copy(img)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(180,120))
            dt = []
            dt.append(img)
            dt = np.array(dt,dtype="uint8")
            dt = dt.reshape(len(dt),120,180,1)
            dt = dt/255
            x = model.predict(dt)
            xl = self.fromScale(1280,x[0][0])
            yl = self.fromScale(720,x[0][1])
            xr = self.fromScale(1280,x[0][2])
            yr = self.fromScale(720,x[0][3])
            while xl < 0:
                xl += 1
            while yl < 0:
                yl += 1
            while xr > 1280:
                xr -= 1
            while yr > 720:
                yr -= 1
            while 1:
                if (yr - yl) == 0 or (xr - xl) == 0:
                    break
                img2 = originalImg[yl:yr,xl:xr]
                cv2.imshow('img',img2)
                k_p = cv2.waitKey(1)
                if k_p == 27:
                    break
            i += 1

    def testModelOnAnyData(self,pth,dataPth):
        model = Sequential()
        model = keras.models.load_model(pth)
        for i in range(0,8):
            img = cv2.imread(os.path.join(dataPth,'img' + str(i)+'.jpg'))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(320,120))
            data = [img]
            data = np.array(data,dtype="uint8")
            data = data.reshape(1,120,320,1)
            x = model.predict(data)
            print(np.argmax(x))



a = NeuralNetwork()
a.anotherRetrievePaths('definitiveTrain')
a.prepareData(True)
a.shuffleData()
a.splitSets(0.30)
# a.retrainModel('handDetector17.h5')
a.newRnModelTrial()
# a.test_data = a.test_data/255
# print(np.argmax(a.test_data))
# print(a.test_data)
# print(len(a.test_data[0][0]))
# print(a.test_data[0])
# a.test_data[0] = (a.test_data[0].reshape(len(a.test_data[0]),-1) - np.mean(a.test_data[0]))/np.std(a.test_data[0])
# print(a.test_data[0])
# print(a.test_labels.reshape(len(a.test_labels),-1))
# print(a.test_labels.shape)
# print(np.mean(a.test_labels))

# a.trainABitMore('handDetector14.h5','poza15.jpg')
# a.trainAndTest('handDetector11.h5','definitive3')
# a.splitSets(0.3)
# a.setTestData('imos2')
# a.train()
# a.loadModel('rnModel2.h5')
# a.testModelOnAnyData('rnModel2.h5','definitiveTest')
# a.plotImage(1)