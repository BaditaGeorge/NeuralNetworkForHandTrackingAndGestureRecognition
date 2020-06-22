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
from keras.callbacks import History
from keras.optimizers import SGD
from keras.optimizers import Adam
import pandas as pd
import os
import cv2
import tensorflow as tf
from PIL import Image
from resizeimage import resizeimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
# tf.compat.v1.disable_eager_execution()

class NueralNetwork:

    def __init__(self):
        self.arr = []
        self.labels = []
        self.data = []
        self.test = []
        self.data_labels = []
        self.test_labels = []
        self.cat = []

    def create_array(self):
        i = 0
        while i < 62053:
            while not os.path.exists(os.path.join('definitiveTrain5','img'+str(i)+'.jpg')):
                i += 1
            self.arr.append(i)
            i += 1

    def read_categorical(self,pth):
        dsc = open(pth,'r')
        line = dsc.readline()
        while line:
            tp = []
            a = line.split(' ')
            for i in a:
                tp.append(int(i))
            self.cat.append(tp)
            line = dsc.readline()
        self.cat = np.asarray(self.cat)
            
        
    def shuffle_array(self):
        ln = len(self.arr)
        for i in range(0,ln):
            x = random.randint(0,ln-1)
            y = random.randint(0,ln-1)
            while x == y:
                x = random.randint(0,ln-1)
                y = random.randint(0,ln-1)
            self.arr[x],self.arr[y] = self.arr[y],self.arr[x]
            tmp = np.copy(self.labels[x])
            self.labels[x] = np.copy(self.labels[y])
            self.labels[y] = tmp

    def shuffle_both_Arrays(self):
        ln = len(self.arr)
        for i in range(0,ln):
            x = random.randint(0,ln-1)
            y = random.randint(0,ln-1)
            while x == y:
                x = random.randint(0,ln-1)
                y = random.randint(0,ln-1)
            self.arr[x],self.arr[y] = self.arr[y],self.arr[x]
            tmp = np.copy(self.labels[x])
            self.labels[x] = np.copy(self.labels[y])
            self.labels[y] = tmp
            tmp = np.copy(self.cat[x])
            self.cat[x] = np.copy(self.cat[y])
            self.cat[y] = tmp

    def create_model2(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(120, 180, 3)))
        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        # model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        # model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        # model.add(Dense(64,activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='relu'))
        sgd = Adam(lr=0.001)
        # sgd = SGD(lr=0.001,momentum=0.3,nesterov=True)
        model.compile(optimizer=sgd,loss='mse',metrics=['mse'])
        self.train_model(model)
        # model.fit(self.train_data,self.train_labels,nb_epoch=10,batch_size=64,validation_data=(self.test_data,self.test_labels),verbose=2)
        # model.save('handDetector23.h5')

    def create_model_categorical2(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(120, 180, 3)))
        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        # model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        # model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        # model.add(Dense(64,activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(6, activation='softmax'))
        sgd = Adam(lr=0.001)
        # sgd = SGD(lr=0.001,momentum=0.3,nesterov=True)
        model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
        self.train_model_categorical(model)
        # model.fit(self.train_data,self.train_labels,nb_epoch=10,batch_size=64,validation_data=(self.test_data,self.test_labels),verbose=2)
        # model.save('handDetector23.h5')
    
    def create_model_categorical(self):
        model = Sequential()
        model.add(Conv2D(64, (1, 1), activation='relu', input_shape=(120, 180, 3)))
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
        model.add(Dense(6, activation='softmax'))
        sgd = Adam(lr=0.001)
        # sgd = SGD(lr=0.001,momentum=0.3,nesterov=True)
        model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
        # model.fit(self.train_data,self.train_labels,nb_epoch=10,batch_size=64,validation_data=(self.test_data,self.test_labels),verbose=2)
        self.train_model_categorical(model)
    
    def create_model_small(self):
        model = Sequential()
        model.add(Conv2D(16, (1, 1), activation='relu', input_shape=(120, 180, 3)))
        model.add(MaxPooling2D((2, 2)))
        # model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        # model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2),strides=(2,2)))
        model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(64,activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='relu'))
        sgd = Adam(lr=0.001)
        # sgd = SGD(lr=0.001,momentum=0.3,nesterov=True)
        model.compile(optimizer=sgd,loss='mse',metrics=['mse'])
        # model.fit(self.train_data,self.train_labels,nb_epoch=10,batch_size=64,validation_data=(self.test_data,self.test_labels),verbose=2)
        self.train_model(model)

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(64, (1, 1), activation='relu', input_shape=(120, 180, 3)))
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
        # model.fit(self.train_data,self.train_labels,nb_epoch=10,batch_size=64,validation_data=(self.test_data,self.test_labels),verbose=2)
        self.train_model(model)
    
    def load_model(self,pth):
        model = Sequential()
        model = keras.models.load_model(pth)
        self.train_model(model)

    def load_Return_model(self,pth):
        model = Sequential()
        model = keras.models.load_model(pth)
        return model

    def read_labels(self,pth):
        dsc = open(pth,'r')
        line = dsc.readline()
        i = 0
        l = 0
        while line:
            arr = line.split(' ')
            r = int(float(arr[2]))
            lX = int(float(arr[0])) - r
            lY = int(float(arr[1])) - r
            rX = int(float(arr[0])) + r
            rY = int(float(arr[1])) + r
            tpl = []
            # while rX >= 1280:
            #     rX -= 1
            # while rY >= 720:
            #     rY -= 1
            tpl.append(lX/1280)
            tpl.append(lY/720)
            tpl.append(rX/1280)
            tpl.append(rY/720)
            # while not os.path.exists(os.path.join('definitiveTrain','img'+str(i)+'.jpg')):
            #     i += 1
            # if lX > 0  and rX > 0 and lY > 0 and rY > 0 and r >0:
            #     print(i)
            #     img = cv2.imread('definitiveTrain/img'+str(i)+'.jpg')
            #     img = img[lY:rY,lX:rX]
            #     while 1:
            #         cv2.imshow('img',img)
            #         kp = cv2.waitKey(1)
            #         if kp == 27:
            #             break
            # i += 1
            self.labels.append(tpl)
            line = dsc.readline()
        self.labels=np.asarray(self.labels)

    def prepareData(self,imgPth):
        # cnt = 0
        img = cv2.imread(imgPth)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(180,120))
        return img
        # fl = open('labels5','r')
        # self.data = np.array(self.data,dtype="uint8")
        # self.data = self.data.reshape(len(self.imgPths),120,180,1)
        # self.data = self.data/255
        # self.labels = np.asarray(self.labels)

    def prepareCut(self,imgPth,data):
        img = cv2.imread(imgPth)
        lY = int(data[1]*720)
        rY = int(data[3]*720)
        lX = int(data[0]*1280)
        rX = int(data[2]*1280)
        if lX < 0:
            lX = 0
        if lY < 0:
            lY = 0
        if rX >= 1280:
            rX = 1279
        if rY >= 720:
            rY = 719
        img = img[lY:rY,lX:rX]
        img = cv2.resize(img,(180,120))
        return img

    def prepareCutPred(self,imgPth,index,model):
        # print('da,aici')
        img = cv2.imread(imgPth)
        cpImg = cv2.resize(img,(180,120))
        arr = [cpImg]
        arr = np.array(arr)
        arr = arr.reshape(1,120,180,3)
        arr = arr/255
        data2 = model.predict(arr)
        lY = int(data2[0][1]*720)
        rY = int(data2[0][3]*720)
        lX = int(data2[0][0]*1280)
        rX = int(data2[0][2]*1280)
        if lX < 0:
            lX = 0
        if lY < 0:
            lY = 0
        if rX >= 1280:
            rX = 1279
        if rY >= 720:
            rY = 719
        # print(lX,lY,rX,rY)
        # img = cv2.resize(img,(180,120))
        if rX-lX <= 0 or rY-lY <= 0:
            lY = int(self.labels[index][1]*720)
            rY = int(self.labels[index][3]*720)
            lX = int(self.labels[index][0]*1280)
            rX = int(self.labels[index][2]*1280)
            # print(lY,rY,lX,rX)
        newImg = img[lY:rY,lX:rX]
        newImg = cv2.resize(newImg,(180,120))
        return newImg

    def train_comp_model(self,model):
        print('comp model')
        epochs = 10
        i_ep = 0
        # history = History()
        btch_sz = 256
        proc = 0.15
        tst_sz = int(len(self.arr)*proc)
        trn_sz = len(self.arr) - tst_sz
        loss = 0.0
        acc = 0.0
        while i_ep < epochs:
            i = 0
            loss = 0
            acc = 0
            print('----TRAIN----')
            while i < trn_sz:
                self.data_labels = []
                self.data = []
                rght = min(btch_sz,trn_sz-i)
                for j in range(i,i+rght):
                    self.data.append(self.prepareData('definitiveTrain5/img' + str(self.arr[j]) + '.jpg'))
                    self.data_labels.append(self.cat[j])
                self.data = np.array(self.data,dtype="uint8")
                self.data = self.data.reshape(len(self.data),120,180,3)
                self.data = self.data/255
                self.data_labels = np.asarray(self.data_labels)
                history = model.fit(self.data,self.data_labels)
                loss += history.history['loss'][0]
                acc += history.history['accuracy'][0]
                print(str(i)+'/'+str(trn_sz))
                i += btch_sz
            print('CompMOdel Loss:'+str(loss)+' -- Acc:'+str(acc))
            loss = 0.0
            mse = 0.0
            i = 0
            print('----TEST----')
            while i < tst_sz:
                self.test = []
                self.test_labels = []
                rght = min(btch_sz,tst_sz-i)
                for j in range(trn_sz+i,trn_sz+i+rght):
                    self.test.append(self.prepareData('definitiveTrain5/img'+str(self.arr[j]) + '.jpg'))
                    self.test_labels.append(self.cat[j])
                self.test = np.array(self.test,dtype="uint8")
                self.test = self.test.reshape(len(self.test),120,180,3)
                self.test = self.test/255
                self.test_labels = np.asarray(self.test_labels)
                res = model.evaluate(self.test,self.test_labels)
                loss += res[0]
                acc += res[1]
                print(str(i) + '/' + str(tst_sz))
                i += btch_sz
            print('CompModel Test Loss:'+str(loss)+' -- Acc:'+str(acc))
            i_ep += 1
        model.save('compNN.h5')

    def train_model_categorical(self,model):
        model_reg = self.load_Return_model('goodHD29.h5')
        epochs = 10
        i_ep = 0
        # history = History()
        dsc = open('catTrain2.txt','a')
        btch_sz = 256
        proc = 0.15
        tst_sz = int(len(self.arr)*proc)
        trn_sz = len(self.arr) - tst_sz
        loss = 0.0
        acc = 0.0
        while i_ep < epochs:
            i = 0
            loss = 0
            acc = 0
            print('----TRAIN----')
            while i < trn_sz:
                self.data_labels = []
                self.data = []
                rght = min(btch_sz,trn_sz-i)
                for j in range(i,i+rght):
                    self.data.append(self.prepareCutPred('definitiveTrain5/img' + str(self.arr[j]) + '.jpg',j,model_reg))
                    self.data_labels.append(self.cat[j])
                self.data = np.array(self.data,dtype="uint8")
                self.data = self.data.reshape(len(self.data),120,180,3)
                self.data = self.data/255
                self.data_labels = np.asarray(self.data_labels)
                history = model.fit(self.data,self.data_labels)
                loss += history.history['loss'][0]
                acc += history.history['accuracy'][0]
                dsc.write(str(history.history['accuracy'][0]) + '\n')
                print(str(i)+'/'+str(trn_sz))
                i += btch_sz
            print('Loss:'+str(loss)+' -- Acc:'+str(acc))
            loss = 0.0
            mse = 0.0
            i = 0
            print('----TEST----')
            while i < tst_sz:
                self.test = []
                self.test_labels = []
                rght = min(btch_sz,tst_sz-i)
                for j in range(trn_sz+i,trn_sz+i+rght):
                    self.test.append(self.prepareCutPred('definitiveTrain5/img'+str(self.arr[j]) + '.jpg',j,model_reg))
                    self.test_labels.append(self.cat[j])
                self.test = np.array(self.test,dtype="uint8")
                self.test = self.test.reshape(len(self.test),120,180,3)
                self.test = self.test/255
                self.test_labels = np.asarray(self.test_labels)
                res = model.evaluate(self.test,self.test_labels)
                loss += res[0]
                acc += res[1]
                # dsc.write(str(res[1]) + '\n')
                print(str(i) + '/' + str(tst_sz))
                i += btch_sz
            print('Test Loss:'+str(loss)+' -- Acc:'+str(acc))
            i_ep += 1
        model.save('catNN3.h5')

    def train_model(self,model):
        epochs = 2
        i_ep = 0
        # history = History()
        btch_sz = 256
        proc = 0.10
        tst_sz = int(len(self.arr)*proc)
        trn_sz = len(self.arr) - tst_sz
        hist_dsc = open('history.txt','a')
        loss = 0.0
        mse = 0.0
        while i_ep < epochs:
            i = 0
            loss = 0
            mse = 0
            print('----TRAIN----')
            while i < trn_sz:
                self.data_labels = []
                self.data = []
                rght = min(btch_sz,trn_sz-i)
                for j in range(i,i+rght):
                    self.data.append(self.prepareData('definitiveTrain5/img' + str(self.arr[j]) + '.jpg'))
                    self.data_labels.append(self.labels[j])
                self.data = np.array(self.data,dtype="uint8")
                self.data = self.data.reshape(len(self.data),120,180,3)
                self.data = self.data/255
                self.data_labels = np.asarray(self.data_labels)
                history = model.fit(self.data,self.data_labels)
                loss += history.history['loss'][0]
                mse += history.history['mse'][0]
                print(str(i)+'/'+str(trn_sz))
                i += btch_sz
            print('Loss:'+str(loss)+' -- mse:'+str(mse))
            hist_dsc.write('Loss:'+str(loss)+' -- mse: ' + str(mse) + '\n')
            loss = 0.0
            mse = 0.0
            i = 0
            print('----TEST----')
            while i < tst_sz:
                self.test = []
                self.test_labels = []
                rght = min(btch_sz,tst_sz-i)
                for j in range(trn_sz+i,trn_sz+i+rght):
                    self.test.append(self.prepareData('definitiveTrain5/img'+str(self.arr[j]) + '.jpg'))
                    self.test_labels.append(self.labels[j])
                self.test = np.array(self.test,dtype="uint8")
                self.test = self.test.reshape(len(self.test),120,180,3)
                self.test = self.test/255
                self.test_labels = np.asarray(self.test_labels)
                res = model.evaluate(self.test,self.test_labels)
                loss += res[0]
                mse += res[1]
                print(str(i) + '/' + str(tst_sz))
                i += btch_sz
            print('Test Loss:'+str(loss)+' -- Test MSE:'+str(mse))
            self.shuffle_array()
            i_ep += 1
        model.save('goodHD32.h5')

    def validateData(self):
        for i in range(0,len(self.arr)):
            xL = int(self.labels[i][0]*1280)
            yL = int(self.labels[i][1]*720)
            xR = int(self.labels[i][2]*1280)
            yR = int(self.labels[i][3]*720)
            img = cv2.imread('definitiveTrain5/img'+str(self.arr[i])+'.jpg')
            # img =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            print(self.arr[i])
            print(xL,yL,xR,yR)
            if xL > 0 and yL > 0 and xR > 0 and yR > 0 and (xR - xL) > 0:
                img = img[yL:yR,xL:xR]
                while 1:
                    cv2.imshow('img',img)
                    kp = cv2.waitKey(1)
                    if kp == 27:
                        break

    def validateDate_Cat(self):
        model = Sequential()
        model = keras.models.load_model('catNN.h5')
        model_reg = self.load_Return_model('goodHD29.h5')
        for i in range(0,len(self.arr)):
            xL = int(self.labels[i][0]*1280)
            yL = int(self.labels[i][1]*720)
            xR = int(self.labels[i][2]*1280)
            yR = int(self.labels[i][3]*720)
            img = cv2.imread('definitiveTrain5/img'+str(self.arr[i])+'.jpg')
            # img =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            dimg = cv2.resize(img,(180,120))
            b = []
            b.append(dimg)
            b = np.array(b)
            b = b.reshape(1,120,180,3)
            b = b/255
            arrOfVals = model_reg.predict(b)
            print(self.arr[i])
            print(xL,yL,xR,yR)
            xL = int(self.labels[0][0]*1280)
            yL = int(self.labels[0][1]*720)
            xR = int(self.labels[0][2]*1280)
            yR = int(self.labels[0][3]*720)
            # print(xL,yL,xR,yR)
            if xL > 0 and yL > 0 and xR > 0 and yR > 0 and (xR - xL) > 0:
                img = self.prepareCutPred('definitiveTrain5/img'+str(self.arr[i])+'.jpg',i,model_reg)
                # print(len(img))
                # if len(img) == 0:
                #     continue
                print(self.cat[i])
                # dtimg = cv2.resize(img,(180,120))
                # a = [dtimg]
                # a = np.asarray(a)
                # a = a.reshape(len(a),120,180,3)
                # a = a/255
                # print(np.argmax(model.predict(a)[0]))
                while 1:
                    cv2.imshow('img',img)
                    kp = cv2.waitKey(1)
                    if kp == 27:
                        break

    def findWorstFit(self,pth):
        model_Reg = self.load_Return_model(pth)
        arrg = []
        labs = []
        cat = []
        for i in range(0,1000):
            lY = int(self.labels[i][1]*720)
            rY = int(self.labels[i][3]*720)
            lX = int(self.labels[i][0]*1280)
            rX = int(self.labels[i][2]*1280)
            dt = []
            dt.append(self.prepareData('definitiveTrain5/img' + str(self.arr[i]) + '.jpg'))
            dt = np.array(dt,dtype="uint8")
            dt = dt.reshape(len(dt),120,180,3)
            dt = dt/255
            arrOfVals = model_Reg.predict(dt)
            lYp = int(arrOfVals[0][1]*720)
            rYp = int(arrOfVals[0][3]*720)
            lXp = int(arrOfVals[0][0]*1280)
            rXp = int(arrOfVals[0][2]*1280)
            sum = abs(lY-lYp) + abs(rY-rYp) + abs(lX-lXp) + abs(rX-rXp)
            print(i)
            if sum > 70:
                arrg.append(self.arr[i])
                labs.append(self.labels[i])
                # cat.append(self.cat[i])
        print(len(arrg),len(labs),len(cat))
        print(len(self.arr),len(self.labels),len(self.cat))
        # self.arr = np.copy(arrg)
        # self.labels = np.copy(labs)
        # self.cat = np.copy(cat)
        print(len(self.arr),len(self.labels),len(self.cat))
        # self.train_model(model_Reg)


a = NueralNetwork()
a.create_array()
a.read_labels('label7')
# a.read_labels('label7')
# a.shuffle_array()
print('ish')
a.read_categorical('finCat')
print('ish')
a.shuffle_both_Arrays()
# print(a.labels[12])
# print(len(a.arr))
# print(len(a.labels))
# a.create_model_small()
a.findWorstFit('goodHD32.h5')
# a.findWorstFit('goodHD30.h5')
# a.validateDate_Cat()
# a.create_model_categorical()
# a.create_model_categorical()
# a.validateData()
# print(a.labels[1])
# a.create_model2()
# a.load_model('goodHD32.h5')
# print(len(a.labels))
# a.train_model(None)