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
from pynput.keyboard import Key,Controller
# from pynput.mouse import Button,Controller

class WebCamHandDetector:
    def __init__(self):
        self.model = None
        self.model_Cat = None
        self.gesture = 5
        self.counters = [0,0,0,0,0]

    def loadModel(self,pth):
        self.model = Sequential()
        self.model = keras.models.load_model(pth)

    def loadModelCat(self,pth):
        self.model_Cat = Sequential()
        self.model_Cat = keras.models.load_model(pth)

    def fromScale(self,mX,v):
        return int(mX*v)
    
    def openWebCam(self):
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        if vc.isOpened():
            rval,frame = vc.read()
        else:
            rval = False
        while rval:
            rval,frame = vc.read()
            # img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(frame,(180,120))
            data = []
            data.append(img)
            data = np.array(data,dtype="uint8")
            data = data.reshape(1,120,180,3)
            data = data/255
            x = self.model.predict(data)
            xl = self.fromScale(640,x[0][0])
            yl = self.fromScale(480,x[0][1])
            xr = self.fromScale(640,x[0][2])
            yr = self.fromScale(480,x[0][3])
            # print(xl,yl,xr,yr)
            frame = cv2.rectangle(frame,(xl,yl),(xr,yr),(255,0,0),2)
            if xl>0 and yl>0 and xr > xl and yr > yl:
                dtimg = frame[yl:yr,xl:xr]
                dt = []
                dtimg = cv2.resize(dtimg,(180,120))
                dt.append(dtimg)
                dt = np.asarray(dt)
                dt = dt.reshape(1,120,180,3)
                dt = dt/255
                print([np.argmax(self.model_Cat.predict(dt)[0]),np.argmax(self.model_Cat.predict(dt)[0]),np.argmax(self.model_Cat.predict(dt)[0])])
            cv2.imshow("preview",frame)
            key = cv2.waitKey(5)
            if key == 27:
                break
        cv2.destroyWindow("preview")

    def gestureRecognition(self):
        ok = 0
        keyboard = Controller()
        cv2.namedWindow("preview")
        # os.system("start C:\\Users\\GoguSpoder\\Desktop\\doc_Licenta_v2.docx")
        vc = cv2.VideoCapture(0)
        if vc.isOpened():
            rval,frame = vc.read()
        else:
            rval = False
        while rval:
            rval,frame = vc.read()
            xl = 120
            yl = 200
            xr = 360
            yr = 440
            imgArea = frame[yl:yr,xl:xr]
            imgArea = cv2.resize(imgArea,(180,120))
            frame = cv2.rectangle(frame,(xl,yl),(xr,yr),(255,0,0),2)
            dt = [imgArea]
            dt = np.array(dt)
            dt = dt.reshape(1,120,180,3)
            dt = dt/255
            value = np.argmax(self.model_Cat.predict(dt))
            if self.gesture != value:
                self.gesture = value
                self.counters = [0,0,0,0,0]
                # if self.gesture == 0:
                #     keyboard.press('s')
                # elif self.gesture == 1:
                #     keyboard.press('a')
                # elif self.gesture == 2:
                #     keyboard.press('l')
                # elif self.gesture == 3:
                #     keyboard.press('u')
                # elif self.gesture == 4:
                #     keyboard.press('t')
            if value < 5:
                self.counters[value] += 1
            if sum(self.counters) == 10:
                i = np.argmax(self.counters)
                if i == 0:
                    keyboard.press('s')
                elif i == 1:
                    keyboard.press('a')
                elif i == 2:
                    keyboard.press('l')
                elif i == 3:
                    keyboard.press('u')
                elif i == 4:
                    keyboard.press('t')
            print(value)
            cv2.imshow("preview",frame)
            key = cv2.waitKey(5)
            if key == 27:
                break
        cv2.destroyWindow("preview")

    def gestureForPpt(self):
        ok = 0
        keyboard = Controller()
        cv2.namedWindow("preview")
        # os.system("start C:\\Users\\GoguSpoder\\Desktop\\doc_Licenta_v2.docx")
        vc = cv2.VideoCapture(0)
        if vc.isOpened():
            rval,frame = vc.read()
        else:
            rval = False
        while rval:
            rval,frame = vc.read()
            xl = 120
            yl = 200
            xr = 360
            yr = 440
            imgArea = frame[yl:yr,xl:xr]
            imgArea = cv2.resize(imgArea,(180,120))
            frame = cv2.rectangle(frame,(xl,yl),(xr,yr),(255,0,0),2)
            dt = [imgArea]
            dt = np.array(dt)
            dt = dt.reshape(1,120,180,3)
            dt = dt/255
            value = np.argmax(self.model_Cat.predict(dt))
            if self.gesture != value:
                self.gesture = value
                self.counters = [0,0,0,0,0]
                # if self.gesture == 0:
                #     keyboard.press('s')
                # elif self.gesture == 1:
                #     keyboard.press('a')
                # elif self.gesture == 2:
                #     keyboard.press('l')
                # elif self.gesture == 3:
                #     keyboard.press('u')
                # elif self.gesture == 4:
                #     keyboard.press('t')
            if value < 5:
                self.counters[value] += 1
            if sum(self.counters) == 10:
                i = np.argmax(self.counters)
                if i == 0:
                    os.system("start C:\\Users\\GoguSpoder\\Desktop\\prezentare.pptx")
                elif i == 1:
                    keyboard.press(Key.down)
                elif i == 2:
                    keyboard.press(Key.up)
                elif i == 3:
                    print(i)
                    # keyboard.press(Key.enter)
                elif i == 4:
                    print(i)
                    # keyboard.press(Key.backspace)
            print(value)
            cv2.imshow("preview",frame)
            key = cv2.waitKey(5)
            if key == 27:
                break
        cv2.destroyWindow("preview")

    def handTracking(self):
        mC = Controller()
        # mB = Button()
        mouseX = -1
        mouseY = -1
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        if vc.isOpened():
            rval,frame = vc.read()
        else:
            rval = False
        while rval:
            rval,frame = vc.read()
            # img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # frm = frame[120:400,80:400]
            img = cv2.resize(frame,(180,120))
            data = []
            data.append(img)
            data = np.array(data)
            data = data.reshape(1,120,180,3)
            data = data/255
            x = self.model.predict(data)
            xl = self.fromScale(640,x[0][0])
            yl = self.fromScale(480,x[0][1])
            xr = self.fromScale(640,x[0][2])
            yr = self.fromScale(480,x[0][3])
            # print(xl,yl,xr,yr)
            frame = cv2.rectangle(frame,(xl-50,yl-20),(xr+20,yr+20),(255,0,0),2)
            # frame = cv2.rectangle(frame,(0,0),(40,40),(255,114,36),2)
            if xl>0 and yl>0 and xr > xl and yr > yl:
                yl -= 20
                yr += 20
                xl -= 50
                xr += 20
                if yl < 0:
                    yl = 0
                if xl < 0:
                    xl = 0
                if yr >= 480:
                    yr = 479
                if xr >= 640:
                    xr = 639
                dtimg = frame[yl:yr,xl:xr]
                dt = []
                dtimg = cv2.resize(dtimg,(180,120))
                dt.append(dtimg)
                dt = np.asarray(dt)
                dt = dt.reshape(1,120,180,3)
                dt = dt/255
                value = np.argmax(self.model_Cat.predict(dt)[0])
                print(value)
                # if self.gesture != value:
                #     self.gesture = value
                #     self.counters = [0,0,0,0,0]
                if value == 5:
                    self.counter = [0,0,0,0,0]
                if value < 5:
                    self.counters[value] += 1
                if value != 5:
                    i = np.argmax(self.counters)
                    if mouseX != -1 and mouseY != -1:
                        xC = int((xl+xr)/2)
                        yC = int((yl+yr)/2)
                        xC = int((xC/640)*1920)
                        yC = int((yC/480)*1080)
                        difX = 0
                        difY = 0
                        if xC > mouseX:
                            difX = 200
                        if xC < mouseX:
                            difX = -200
                        if yC > mouseY:
                            difY = 200
                        if yC < mouseY:
                            difY = -200
                        mouseX += difX
                        mouseY += difY
                        mC.move(difX,difY)
                    else:
                        xC = int((xl+xr)/2)
                        yC = int((yl+yr)/2)
                        pxC = xC/640
                        pyC = yC/480
                        mC.position = (500,500)
                        mouseX = 500
                        mouseY = 500
                # print([np.argmax(self.model_Cat.predict(dt)[0]),np.argmax(self.model_Cat.predict(dt)[0]),np.argmax(self.model_Cat.predict(dt)[0])])
            cv2.imshow("preview",frame)
            key = cv2.waitKey(5)
            if key == 27:
                break
        cv2.destroyWindow("preview")

    def controlByHT(self):
        # mC = Controller()
        # mB = Button()
        keyboard = Controller()
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        if vc.isOpened():
            rval,frame = vc.read()
        else:
            rval = False
        while rval:
            rval,frame = vc.read()
            # img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # frm = frame[120:400,80:400]
            img = cv2.resize(frame,(180,120))
            data = []
            data.append(img)
            data = np.array(data)
            data = data.reshape(1,120,180,3)
            data = data/255
            x = self.model.predict(data)
            xl = self.fromScale(640,x[0][0])
            yl = self.fromScale(480,x[0][1])
            xr = self.fromScale(640,x[0][2])
            yr = self.fromScale(480,x[0][3])
            # print(xl,yl,xr,yr)
            frame = cv2.rectangle(frame,(xl-50,yl-20),(xr+20,yr+20),(255,0,0),2)
            # frame = cv2.rectangle(frame,(0,0),(40,40),(255,114,36),2)
            if xl>0 and yl>0 and xr > xl and yr > yl:
                yl -= 20
                yr += 20
                xl -= 50
                xr += 20
                if yl < 0:
                    yl = 0
                if xl < 0:
                    xl = 0
                if yr >= 480:
                    yr = 479
                if xr >= 640:
                    xr = 639
                dtimg = frame[yl:yr,xl:xr]
                dt = []
                dtimg = cv2.resize(dtimg,(180,120))
                dt.append(dtimg)
                dt = np.asarray(dt)
                dt = dt.reshape(1,120,180,3)
                dt = dt/255
                value = np.argmax(self.model_Cat.predict(dt)[0])
                print(value)
                if self.gesture != value:
                    self.gesture = value
                    self.counters = [0,0,0,0,0]
                if value < 5:
                    self.counters[value] += 1
                if sum(self.counters) == 23:
                    i = np.argmax(self.counters)
                    if i == 0:
                        os.system("start C:\\Users\\GoguSpoder\\Desktop\\prezentare.pptx")
                    elif i == 1:
                        keyboard.press(Key.f5)
                        print(i)
                    elif i == 2:
                        keyboard.press(Key.esc)
                        print(i)
                    elif i == 3:
                        print(i)
                        keyboard.press(Key.right)
                    elif i == 4:
                        print(i)
                        keyboard.press(Key.left)
                # if value == 5:
                #     self.counter = [0,0,0,0,0]
                # if value < 5:
                #     self.counters[value] += 1
                # if value != 5:
                #     i = np.argmax(self.counters)
                #     if mouseX != -1 and mouseY != -1:
                #         xC = int((xl+xr)/2)
                #         yC = int((yl+yr)/2)
                #         xC = int((xC/640)*1920)
                #         yC = int((yC/480)*1080)
                #         difX = 0
                #         difY = 0
                #         if xC > mouseX:
                #             difX = 200
                #         if xC < mouseX:
                #             difX = -200
                #         if yC > mouseY:
                #             difY = 200
                #         if yC < mouseY:
                #             difY = -200
                #         mouseX += difX
                #         mouseY += difY
                #         mC.move(difX,difY)
                #     else:
                #         xC = int((xl+xr)/2)
                #         yC = int((yl+yr)/2)
                #         pxC = xC/640
                #         pyC = yC/480
                #         mC.position = (500,500)
                #         mouseX = 500
                #         mouseY = 500
                # print([np.argmax(self.model_Cat.predict(dt)[0]),np.argmax(self.model_Cat.predict(dt)[0]),np.argmax(self.model_Cat.predict(dt)[0])])
            cv2.imshow("preview",frame)
            key = cv2.waitKey(5)
            if key == 27:
                break
        cv2.destroyWindow("preview")


a = WebCamHandDetector()
a.loadModel('goodHD29.h5')
a.loadModelCat('catNN3.h5')
# a.openWebCam()
# a.controlByHT()
# a.handTracking()
a.gestureRecognition()
# a.gestureForPpt()