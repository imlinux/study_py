import cv2
import numpy as np

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


img_raw = cv2.imread("/home/dong/tmp/score/img/0/SHENJIANG_F_00028.pdf-1.jpg")
img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

(thresh, img_bin) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

img_bin = 255-img

kernel_length = np.array(img).shape[1]//40
verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)


img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

alpha = 0.5
beta = 1.0 - alpha

img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)

(thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

(contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")


idx = 0
for c in contours:
    # Returns the location and width,height for every contour
    x, y, w, h = cv2.boundingRect(c)

    # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
    if 80 < w < 170 and 80 < h < 170:
        idx += 1
        new_img = img_raw[y:y + h, x:x + w]
        cv2.imwrite("output/" + str(idx) + '.png', new_img)

cv2.imwrite("t.jpg", img_final_bin)
cv2.drawContours(img_raw, contours, -1, (0, 0, 255), 3)
cv2.imwrite("t1.jpg", img_raw)
cv2.imwrite("verticle_lines_img.jpg", verticle_lines_img)
cv2.imwrite("horizontal_lines_img.jpg", horizontal_lines_img)