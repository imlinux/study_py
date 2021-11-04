import cv2
import numpy as np
import os
from my_tools import extract_img_from_pdf

def sort_contours(cnts):

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    return sorted(zip(cnts, boundingBoxes), key=lambda b: (b[1][1], b[1][0]), reverse=False)


def split_img(img_raw):

    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    thresh, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255-img_bin

    kernel_length = np.array(img_bin).shape[1]//40
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    verticle_lines_img = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.morphologyEx(verticle_lines_img, cv2.MORPH_CLOSE, verticle_kernel, iterations=100)

    horizontal_lines_img = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.morphologyEx(horizontal_lines_img, cv2.MORPH_CLOSE, hori_kernel, iterations=100)


    alpha = 0.5
    beta = 1.0 - alpha

    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)

    _, img_final_bin = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    idx = 0
    for c, b in sort_contours(contours):

        x, y, w, h = b

        if 80 < w < 600 and 80 < h < 600:
            idx += 1
            new_img = img_raw[y:y + h, x:x + w]

            os.makedirs("output", exist_ok=True)
            cv2.imwrite("output/" + str(idx) + '.png', new_img)

    cv2.imwrite("t.jpg", img_final_bin)
    cv2.drawContours(img_raw, contours, -1, (0, 0, 255), 3)
    cv2.imwrite("t1.jpg", img_raw)
    cv2.imwrite("verticle_lines_img.jpg", verticle_lines_img)
    cv2.imwrite("horizontal_lines_img.jpg", horizontal_lines_img)


imgs = extract_img_from_pdf("/home/dong/tmp/score/SHENJIANG_F_00032.pdf", 9)
for img in imgs:
    split_img(img)
