import cv2
import numpy as np


img_path = '/home/dong/tmp/SHENJIANG_F_00028.pdf/0.jpg'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
_, img = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[1]

cv2.drawContours(img, contours, -1, (0,255,0), 50)

cv2.imshow("", img)
cv2.waitKey(0)