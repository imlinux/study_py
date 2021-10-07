import cv2
import numpy as np

img = cv2.imread("/home/dong/tmp/zuowen/img/0/JUYE_F_00007.pdf-1.jpg")

yansemask = img[:,:,0] + img[:,:,1] + img[:,:,2]
img[yansemask == 0] = 255
#img[-250:, 100:2000] = [255, 255, 255]
img[np.where((img==[0,0,255]).all(axis=2))] = [0,0,0]
cv2.imwrite("/home/dong/tmp/tmp.jpg", img)