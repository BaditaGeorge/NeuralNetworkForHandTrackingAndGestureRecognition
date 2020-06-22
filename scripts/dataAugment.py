import cv2
import random
import numpy as np
import os
from scipy import fftpack

class AugmentData:

    def compute(self,x,y,r):
        xL = x - r
        yL = y - r
        xR = x + r
        yR = y + r
        return [xL,yL,xR,yR]

    def randomizeBackground(self,pth):
        cx = 450
        cy = 372
        r = 194
        img = cv2.imread(pth)
        [xL,yL,xR,yR] = self.compute(cx,cy,r)
        for i in range(0,len(img)):
            for j in range(0,len(img[i])):
                if (j < xL or j > xR) or (i < yL or i > yR):
                    img[i][j][0] = random.randint(0,255)
                    img[i][j][1] = random.randint(0,255)
                    img[i][j][2] = random.randint(0,255)
        cv2.imwrite('modif.jpg',img)

    def median_filter(self,data):
        temp = []
        for i in range(0,len(data)):
            for j in range(0,len(data[0])):
                temp = []
                if i + 3 < len(data) and j + 3 < len(data[0]):
                    for t in range(0,3):
                        for k in range(0,3):
                            temp.append([data[i+t][j+k][0],data[i+t][j+k][1],data[i+t][j+k][2]])
                    R = 0
                    G = 0
                    B = 0
                    for t in range(0,len(temp)):
                        R += temp[t][0]
                        G += temp[t][1]
                        B += temp[t][2]
                    data[i][j] = [R//9,G//9,B//9]
        return data

    def gaussian_noise(self,data,var):
        mean = 0.93
        sigma = var ** 0.5
        gaussian = np.random.normal(mean,sigma,(720,1280))
        # data = data + gaussian
        data2 = np.copy(data)
        noisy_image = np.zeros(data.shape)
        # for i in range(0,len(data)):
        #     for j in range(0,len(data[0])):
        #         noisy_image[i][j][0] = data[i][j][0] + gaussian[i][j]
        #         noisy_image[i][j][1] = data[i][j][1] + gaussian[i][j]
        #         noisy_image[i][j][2] = data[i][j][2] + gaussian[i][j]
        noisy_image[:,:,0] = data[:,:,0] + gaussian
        noisy_image[:,:,1] = data[:,:,1] + gaussian
        noisy_image[:,:,2] = data[:,:,2] + gaussian
        cv2.normalize(noisy_image,noisy_image,0,255,cv2.NORM_MINMAX,dtype=-1)
        noisy_image = noisy_image.astype(np.uint8)
        return noisy_image

    def saltPapper(self,img):
        quant = 10000
        i = 0
        cap1 = 720
        cap2 = 1280
        while i < quant:
            x = np.random.randint(0,cap1)
            y = np.random.randint(0,cap2)
            img[x,y,:] = 0
            x = np.random.randint(0,cap1)
            y = np.random.randint(0,cap2)
            img[x,y,:] = 255
            i += 1
        return img

    def blurImg(self,pth):
        img = cv2.imread(pth)
        img = cv2.blur(img,(25,25))
        print(img[0][0])
        while 1:
            cv2.imshow('img',img)
            kp = cv2.waitKey(1)
            if kp == 27:
                break

    def invertColor(self,img):
        img = cv2.bitwise_not(img)
        return img

    def invertColor2(self,pth):
        img = cv2.imread(pth)
        for i in range(0,img.shape[0]):
            for j in range(0,img.shape[1]):
                img[i,j,:] = abs(img[i,j,:] - 255)
        while 1:
            cv2.imshow('img',img)
            kp = cv2.waitKey(1)
            if kp == 27:
                break

    def sp_noise(self,img,prob):
        # img = cv2.imread(pth)
        out = np.zeros(img.shape,np.uint8)
        for i in range(0,img.shape[0]):
            for j in range(0,img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    out[i][j] = 128
                    for k in range(5):
                        out[i-k][j-k] = 128 + 10*rdn
                else:
                    out[i][j] = img[i][j]
        return out

a = AugmentData()
# a.saltPapper('definitiveTrain/img12.jpg')
# a.blurImg('definitiveTrain/img12.jpg')
# a.invertColor('definitiveTrain/img12.jpg')
# a.sp_noise('definitiveTrain/img12.jpg',0.07)
i = 0
j = 49646
# img = cv2.imread('definitiveTrain/img12.jpg')
kernel = np.ones((9,9),np.uint8)
# img = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
while i < 12426:
    while not os.path.exists(os.path.join('definitiveTrain4','img'+str(i)+'.jpg')):
        i += 1
    img = cv2.imread(os.path.join('definitiveTrain4','img'+str(i)+'.jpg'))
    img = a.gaussian_noise(img,1100)
    img = cv2.erode(img,kernel,iterations=1)
    img = cv2.dilate(img,kernel,iterations=1)
    cv2.imwrite(os.path.join('definitiveTrain5','img'+str(j)+'.jpg'),img)
    i += 1
    j += 1
# img = cv2.imread('definitiveTrain/img12.jpg')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img = cv2.imread('definitiveTrain/img12.jpg')
# img = a.gaussian_noise(img,995)
# while 1:
#     cv2.imshow('img',img)
#     kp = cv2.waitKey(1)
#     if kp == 27:
#         break
# i = 0
# var = 995
# desc = open('labels5','r')
# while i < 10812:
#     while not os.path.exists(os.path.join('definitiveTrain','img'+str(i)+'.jpg')):
#         i += 1
#     if i % 1280 == 0:
#         var += 17
#     img = cv2.imread('definitiveTrain/img'+str(i)+'.jpg')
#     imgi = a.gaussian_noise(img,var,desc.readline())
#     cv2.imwrite('definitiveTrain2/img'+str(i)+'.jpg',imgi)
#     i += 1

# a.randomizeBackground('definitiveTrain/img12.jpg')