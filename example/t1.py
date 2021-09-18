import cv2
import numpy as np

img = cv2.imread("/home/dong/tmp/mnist/8/train17.jpg", 0)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, tmp = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
#
# tmp = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 15)
# tmp = cv2.blur(tmp, (5,5), 0)
# ret3,tmp = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((5,5), np.uint8)
img = cv2.erode(img, kernel, iterations=1)
cv2.imshow("", img)
cv2.waitKey(0)