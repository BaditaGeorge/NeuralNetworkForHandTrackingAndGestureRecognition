import os
from PIL import Image
from resizeimage import resizeimage
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self):
        print('created')
        self.imageArray = []
        self.labels = []
    
    def loadImage(self,path):
        for (dirpath,dirnames,filenames) in os.walk(path):
            for filename in filenames:
                # print(os.path.join(dirpath,filename))
                img = Image.open(os.path.join(dirpath,filename))
                self.imageArray.append(img)
                i = 0
                while filename[i] >= 'a' and filename[i] <='z':
                    i += 1
                self.labels.append(filename[:i])
    
    def resizeImages(self,startIndex=None,endIndex=None):
        if startIndex == None or endIndex == None:
            startIndex = 0
            endIndex = len(self.imageArray)
        if endIndex < startIndex:
            return
        if endIndex <= len(self.imageArray):
            for i in range(startIndex,endIndex):
                self.imageArray[i] = resizeimage.resize_cover(self.imageArray[i],[50,50],validate=False)

    def saveImages(self,path,startIndex=None,endIndex=None):
        if startIndex == None or endIndex == None:
            startIndex = 0
            endIndex = len(self.imageArray)

        if not os.path.exists(path):
            os.mkdir(path)

        if endIndex < startIndex:
            return

        if endIndex <= len(self.imageArray):
            for i in range(startIndex,endIndex):
                self.imageArray[i].save(os.path.join(path,'image'+str(i)+'.jpg'))

    def showAt(self,index):
        self.imageArray[index].show()

    def mapForHSVConv(self,img):
        img = np.asarray(img.convert('RGB'))
        img = img[:,:,::-1].copy()
        return img

    def alterSkinPixels(self,index):
        min_HSV = np.array([0,58,30],dtype="uint8")
        max_HSV = np.array([33,255,255],dtype="uint8")
        img = self.mapForHSVConv(self.imageArray[index])
        imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        skinRegHSV = cv2.inRange(imgHSV,min_HSV,max_HSV)
        skinHSV = cv2.bitwise_and(img,img,mask=skinRegHSV)
        # cv2.imwrite('picture1.jpg',skinHSV)
        self.imageArray[index] = Image.fromarray(skinHSV)
        pixs = self.imageArray[index].load()
        for i in range(self.imageArray[index].size[0]):
            for j in range(self.imageArray[index].size[1]):
                if pixs[i,j] != (0,0,0):
                    pixs[i,j] = (255,255,255)

    def alterInRange(self,startIndex=None,endIndex=None):
        if startIndex == None or endIndex == None:
            startIndex = 0
            endIndex = len(self.imageArray)
        for i in range(startIndex,endIndex):
            self.alterSkinPixels(i)

    def addCVImage(self,img):
        self.imageArray.append(Image.fromarray(img))

    def alterPixels(self):
        pixels = self.imageArray[0].load()
        for i in range(0,self.imageArray[0].size[0]):
            for j in range(0,self.imageArray[0].size[1]):
                pixels[i,j] = (0,0,0)
        self.imageArray[0].show()



# ip = ImageProcessor()
# ip.loadImage('./train')
# ip.resizeImages(0,len(ip.imageArray)-1)
# ip.resizeImages(0,10)
# ip.alterPixels()
# cv2.imwrite('picture1.jpg',ip.alterSkinPixels(1))
# ip.resizeImages()
# ip.alterInRange()
# ip.saveImages('poze')
# print(ip.imageArray[11].size[0],ip.imageArray[11].size[1])
# ip.showAt(163)
# ip.showAt(120)