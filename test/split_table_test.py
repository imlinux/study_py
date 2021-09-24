import cv2
import numpy as np

img = cv2.imread("/home/dong/Downloads/gaitubao_table_10.png", cv2.IMREAD_GRAYSCALE)
_,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img = ~img

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[0]//20, 1))
eroded = cv2.erode(img, kernel, iterations = 1)
dilate_row = cv2.dilate(eroded, kernel, iterations = 1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img.shape[0]//20))
eroded = cv2.erode(img, kernel, iterations = 1)
dilate_col = cv2.dilate(eroded, kernel, iterations = 1)

bitwise_and = cv2.bitwise_and(dilate_row, dilate_col)

add = cv2.add(dilate_row, dilate_col)

index_array = np.where(bitwise_and > 0)

d = np.column_stack((index_array[1], index_array[0]))

cv2.imshow("", img)

print(d)
print(np.unique(index_array[0]))



cv2.waitKey(0)