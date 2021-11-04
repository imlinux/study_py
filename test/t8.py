import os

import cv2
import numpy as np
from imutils import perspective
from my_tools import extract_img_from_pdf


def split_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite("t1.jpg", blur)
    edge = cv2.Canny(blur, 100, 200)
    cv2.imwrite("t2.jpg", edge)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        ((x, y), (w, h), angle) = cv2.minAreaRect(c)
        if 1000 < w and 1000 < h:
            cv2.imwrite("t4.jpg", cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3))
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            cv2.imwrite("t3.jpg", cv2.drawContours(img.copy(), [approx], -1, (0, 0, 255), 3))
            if len(approx) == 4:
                t = np.array(approx)
                t = t.reshape(-1, 2)
                print("角度调整")
                return img, perspective.four_point_transform(img, t)
            break
    print("角度保持不变")


for img in extract_img_from_pdf("/home/dong/tmp/zuowen/JUYE_F_00007.pdf", 1):
    img = img[2100:]
    split_img(img)
    break