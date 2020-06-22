import cv2
import os

def openWebCam(toPath):
    if not os.path.exists(toPath):
        os.mkdir(toPath) 
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        rval,frame = vc.read()
    else:
        rval = False
    cnt = 0
    cnt1 = 0
    while rval:
        cv2.imshow("preview",frame)
        rval,frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:
            break
        if key == 103:
            cnt += 1
            cv2.imwrite(toPath + '/hello' + str(cnt) + '.jpg',frame)
        elif key == 98:
            cnt1 += 1
            cv2.imwrite(toPath + '/nohello' + str(cnt1) + '.jpg',frame)

    cv2.destroyWindow("preview")

openWebCam('./train')