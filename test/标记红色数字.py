import cv2
import numpy as np
import imutils


img = cv2.imread("/home/dong/tmp/zuowen/img/0/JUYE_F_00007.pdf-3.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## Gen lower mask (0-5) and upper mask (175-180) of RED
lower_red = np.array([150, 103, 100])
upper_red = np.array([180,255,255])

mask = cv2.inRange(img_hsv, lower_red, upper_red)
img[np.where(mask == 0)] = [0, 0, 0]
img[np.where(mask != 0)] = [255, 255, 255]

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


contours = cv2.findContours(img_gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cnts = contours[0]

for cnt in cnts:
    # 外接矩形框，没有方向角
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imwrite("t.jpg", img)