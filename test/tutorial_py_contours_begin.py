import cv2
import numpy as np


img_path = '/home/dong/tmp/2021-10-01_10-19.png'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#img = ~img[10: -10, 10: -10]

_, img = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

x, y, w, h = cv2.boundingRect(contours[3])

cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("", img)
cv2.waitKeyEx()