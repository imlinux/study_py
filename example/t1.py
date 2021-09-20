import cv2
import numpy as np

img = cv2.imread("/home/dong/tmp/2.jpg", 0)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, tmp = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
#
#tmp = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 15)
img = cv2.blur(img, (5, 5), 0)
ret3,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

img = cv2.resize(img, (28, 28))
cv2.imshow("", img)
cv2.waitKey(0)

cv2.imwrite("/home/dong/tmp/mnist/train/2.jpg", img)