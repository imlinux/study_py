import cv2
import numpy as np

img_path = '/home/dong/tmp/SHENJIANG_F_00028.pdf/0.jpg'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img = ~img

h = 106
w = [246, 348]

columns = [(266, 512), (512, 864), (994, 1160), (1160, 1316), (1472, 1644), (1644,1800), (1942, 2108), (2108, 2270)]
first_row = 565
for i in range(25):
    row = first_row + i * h
    for column in columns:
        cv2.imshow(str(i), img[row: row + h, column[0]: column[1]])
        cv2.waitKey(0)
