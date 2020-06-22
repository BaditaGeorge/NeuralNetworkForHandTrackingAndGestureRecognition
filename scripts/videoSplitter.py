import os
from PIL import Image
from resizeimage import resizeimage
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


class VideoSplitter:
    def __init__(self):
        self.initialX = 0
        self.initialY = 0
        self.finalX = 0
        self.finalY = 0
        self.img = None
        self.originalImg = None
        self.descLab = None
        self.count = 0
        self.defCount = 10812

    def splitInFrames(self,pth,svPth):
        if not os.path.exists(pth):
            return
        if not os.path.exists(svPth):
            os.mkdir(svPth)
        vid = cv2.VideoCapture(pth)
        succ,img = vid.read()
        cnt = 0
        while succ:
            cv2.imwrite(os.path.join(svPth,'img'+str(cnt)+'.jpg'),img)
            succ,img = vid.read()
            cnt += 1

    def labelImage2(self,pth,lblPth):
        img = cv2.imread(os.path.join(pth,'img'+str(self.count)+'.jpg'))
        dsc = open(lblPth,'a')
        while self.count < 20000:
            if os.path.exists(os.path.join(pth,'img'+str(self.count)+'.jpg')):
                img = cv2.imread(os.path.join(pth,'img'+str(self.count)+'.jpg'))
                cv2.imshow('img',img)
                k_p = cv2.waitKey(1)
                arr = [0,0,0,0,0,0]
                if k_p == 27:
                    break
                elif k_p == ord('1') or k_p == ord('2') or k_p == ord('3') or k_p == ord('4') or k_p == ord('5') or k_p == ord('6'):
                    self.count += 1
                    arr[k_p-ord('0')-1] = 1
                    dsc.write((' '.join(str(el) for el in arr))+'\n')
            else:
                self.count += 1


    def labelImage(self,pth,lblPth,pthToSave):
        def clbFun(event,x,y,flags,params):
            if event == cv2.EVENT_RBUTTONDOWN:
                self.img = np.copy(self.originalImg)
            
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x,y)
                self.initialX = x
                self.initialY = y

            if event == cv2.EVENT_LBUTTONUP:
                print(x,y)
                self.finalX = x
                self.finalY = y
                cv2.line(self.img,(self.initialX,self.initialY),(self.finalX,self.finalY),(0,255,0),thickness=2)
                cv2.circle(self.img,(self.initialX,self.initialY),abs(self.finalY - self.initialY),(0,255,0),thickness=2)
                # for i in range(self.initialY,self.finalY):
                    # self.pixels[self.initialX,i] = [0,0,0]

        img = cv2.imread(os.path.join(pth,'img'+str(self.count)+'.jpg'))
        cv2.namedWindow('image')
        self.img = img
        self.originalImg = np.copy(img)
        cv2.setMouseCallback('image',clbFun)
        while 1:
            cv2.imshow('image',self.img)
            key_pressed = cv2.waitKey(1)
            if key_pressed == 27:
                break
            elif key_pressed == ord('s'):
                self.descLab = open(lblPth,'a')
                if(self.initialX == None and self.initialY == None):
                    self.descLab.write('0 0 0.0\n')
                else:
                    self.descLab.write(str(self.initialX) + ' ' + str(self.initialY) + ' ' + str(abs(float(self.finalY - self.initialY))) + '\n')
                self.descLab.close()
                self.initialX = None
                self.initialY = None
                self.finalX = None
                self.finalY = None
                self.count += 1
                cv2.imwrite(os.path.join(pthToSave,'img'+str(self.defCount)+'.jpg'),self.originalImg)
                self.defCount += 1
                self.img = cv2.imread(os.path.join(pth,'img' + str(self.count) + '.jpg'))
                self.originalImg = np.copy(self.img)
            elif key_pressed == ord('c'):
                self.count += 1
                self.img = cv2.imread(os.path.join(pth,'img' + str(self.count) + '.jpg'))
                self.originalImg = np.copy(self.img)
                
        cv2.destroyAllWindows()

    
    def toScale(self,mX,v):
        return v/mX
    
    def fromScale(self,mX,v):
        return int(mX*v)
    
    def cropTrial(self,pthImg):
        img = cv2.imread(pthImg)
        fl = open('labels','r')
        arr = fl.readline().split(' ')
        img = cv2.resize(img,(320,120))
        print(img.shape)
        inX = int(arr[0])
        inY = int(arr[1])
        r = int(float(arr[2][:len(arr[2])-1]))
        tY = self.fromScale(120,self.toScale(720,inY))
        tX = self.fromScale(320,self.toScale(1280,inX))
        tR = self.fromScale(120,self.toScale(720,r)) 
        print(tY,tX,tR)
        print(inX,inY,r)
        img = img[(tY-tR):(tY+tR),(tX-tR):(tX+tR)]
        while 1:
            cv2.imshow('image',img)
            key_pressed = cv2.waitKey(1)
            if key_pressed == 27:
                break

    def cropTrialInBatch(self,pth,lbPth):
        pths = []
        desc = open(lbPth,'r')
        cnt = 0
        while cnt < 6956:
            if os.path.exists(os.path.join(pth,'img'+str(cnt)+'.jpg')):
                desc.readline()
            cnt += 1
        while 1:
            while not os.path.exists(os.path.join(pth,'img'+str(cnt)+'.jpg')):
                cnt += 1
            arr = desc.readline().split(' ')
            x = int(arr[0])
            y = int(arr[1])
            r = int(float(arr[2]))
            if x != 0 and y != 0 and r != 0:
                img = cv2.imread(os.path.join(pth,'img'+str(cnt)+'.jpg'))
                while y-r < 0:
                    r -= 1
                while y+r >= 720:
                    r -= 1
                while x-r < 0:
                    r -= 1
                while x+r >= 1280:
                    r -= 1
                img = img[(y-r):(y+r),(x-r):(x+r)]
                while 1:
                    cv2.imshow('img',img)
                    k_p = cv2.waitKey(1)
                    if k_p == 27:
                        break
            cnt += 1
        
    def correctSamples(self,cat,reg,fin_cat,fin_reg):
        dsc = open(cat,'r')
        dsc2 = open(reg,'r')
        dsc3 = open(fin_cat,'a')
        dsc4 = open(fin_reg,'a')
        line = dsc.readline()
        line2 = dsc2.readline()
        nr = 0
        while line:
            arr = line2.split(' ')
            if (arr[0] == '0' and arr[1] == '0') or arr[2][0] == '0':
                dsc3.write('0 0 0 0 0 1\n')
                x = random.randrange(170,950)
                y = random.randrange(170,600)
                r = float(random.randrange(120,155))
                dsc4.write((' '.join([str(x),str(y),str(r)])) + '\n')
            else:
                dsc3.write(line)
                dsc4.write(line2)
            line = dsc.readline()
            line2 = dsc2.readline()


vs = VideoSplitter()
# vs.correctSamples('categories','label7','finCat','finReg')
# vs.splitInFrames('video8.mp4','caps7')
# vs.labelImage2('definitive6','categories2')
vs.labelImage('caps7','lab','def')
# vs.cropTrial(os.path.join('definitive','img0.jpg'))
# vs.cropTrialInBatch('definitive3','labels3')