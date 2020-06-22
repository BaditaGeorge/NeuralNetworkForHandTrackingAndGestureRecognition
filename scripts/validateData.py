import cv2
import os

dsc = open('labels5','r')
i = 0
while i < 10812:
    while not os.path.exists(os.path.join('definitiveTrain','img'+str(i)+'.jpg')):
        i += 1 
    img = cv2.imread('definitiveTrain/img'+str(i)+'.jpg')
    line = dsc.readline()
    arr = line.split(' ')
    r = int(float(arr[2]))
    xL = int(float(arr[0])) - r
    yL = int(float(arr[1])) - r
    xR = int(float(arr[0])) + r
    yR = int(float(arr[1])) + r
    while xR >= 1280:
        xR -= 1
    while yR >= 720:
        yR -= 1
    if yL > 0 and yR > 0 and xL > 0 and xR > 0 and r > 0:
        print(i)
        img = img[yL:yR,xL:xR]
        while 1:
            cv2.imshow('img',img)
            kp = cv2.waitKey(1) 
            if kp == 27:
                break
    i += 1