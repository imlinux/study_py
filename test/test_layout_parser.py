import layoutparser as lp
import cv2

img = cv2.imread("/home/dong/tmp/SHENJIANG_F_00028.pdf/0.jpg")

model = lp.Detectron2LayoutModel("lp://TableBank/faster_rcnn_R_50_FPN_3x/config")
layout = model.detect(img)