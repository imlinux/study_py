import cv2
import numpy as np
## Read and merge


img = cv2.imread("/home/dong/tmp/1.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## Gen lower mask (0-5) and upper mask (175-180) of RED
lower_red = np.array([150, 103, 100])
upper_red = np.array([180,255,255])

mask = cv2.inRange(img_hsv, lower_red, upper_red)
img[np.where(mask > 0)] = [255, 255, 255]

cv2.imwrite("t.jpg", img)