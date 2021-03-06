import os

import cv2
import numpy as np
from imutils import perspective

from my_tools import extract_img_from_pdf
from paddleocr import PPStructure,draw_structure_result,save_structure_res


table_engine = PPStructure(show_log=True, use_gpu=False)

def sort_contours(cnts):
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    return sorted(zip(cnts, boundingBoxes), key=lambda b: (b[1][1], b[1][0]), reverse=False)


def edge_detection(image):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 100, 200)
    cv2.imwrite("t2.jpg", edge)
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        ((x, y), (w, h), angle) = cv2.minAreaRect(c)
        if 1000 < w and 1000 < h:
            cv2.imwrite("t4.jpg", cv2.drawContours(image.copy(), contours, -1, (0, 0, 255), 3))
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            cv2.imwrite("t3.jpg", cv2.drawContours(image.copy(), [approx], -1, (0, 0, 255), 3))
            if len(approx) == 4:
                t = np.array(approx)
                t = t.reshape(-1, 2)
                print("角度调整")
                return image, perspective.four_point_transform(image, t)
    print("角度保持不变")
    return image, image


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


for img in extract_img_from_pdf("/home/dong/tmp/score/SHENJIANG_F_00032.pdf", 9):
    result = table_engine(img)
    img = result[0]["img"]
    img, table = edge_detection(img)
    split_img(table)
    break
