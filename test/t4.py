import cv2

import numpy as np

image=cv2.imread(r"/home/dong/tmp/tmp1.jpg")

np.set_printoptions(threshold=np.inf)

hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

low_range = np.array([150, 103, 100])

high_range = np.array([180, 255, 255])

th = cv2.inRange(hue_image, low_range, high_range)

index1 = th == 255

img = np.zeros(image.shape, np.uint8)

img[:, :] = (255,255,255)

img[index1] = image[index1]#(0,0,255)

gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

kernel = np.ones((5, 5), np.uint8)

gray = cv2.dilate(~gray, kernel, iterations=4)

contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

tmp3 = image.copy()

def cnt_area(cnt):

    area = cv2.contourArea(cnt)

    return area

contours.sort(key = cnt_area, reverse=True)

cnt = contours[0]

x, y, w, h = cv2.boundingRect(cnt)

red=image[y:y+h,x-10:x+w+10]#因为图片特性左右加了10个像素，不加也是可以的，只是为了方便演示

cv2.imwrite("red.jpg",red)

b,g,r=cv2.split(red)

redcopy=red.copy()

ret,th2 = cv2.threshold(r,160,255,cv2.THRESH_BINARY)

red[:,:,0] = th2

red[:,:,1] = th2

red[:,:,2] = th2

tmp3[y:y+h,x-10:x+w+10]=red

cv2.imwrite("t.jpg",tmp3)