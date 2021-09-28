import cv2
import numpy as np


def split_to_row(index_array):
    d = np.column_stack((index_array[1], index_array[0]))

    row_item = []
    row = -1
    for (x, y) in d:
        if row == -1:
            row = y

        if y != row:
            row = y
            yield row_item
            row_item = []

        row_item.append((x, y))


def split_row_to_cell(row):
    l = list(row)
    for index, _ in enumerate(l):
        if index == 0: continue

        row1 = l[index - 1]
        row2 = l[index]

        for col_index in range(min(len(row1), len(row2))):
            if col_index == 0: continue
            yield row1[col_index - 1], row1[col_index], row2[col_index - 1], row2[col_index]


def main():
    img = cv2.imread("/home/dong/tmp/SHENJIANG_F_00028.pdf/0.jpg", cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = ~img

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[0] // 20, 1))
    eroded = cv2.erode(img, kernel, iterations=1)
    dilate_row = cv2.dilate(eroded, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img.shape[0] // 20))
    eroded = cv2.erode(img, kernel, iterations=1)
    dilate_col = cv2.dilate(eroded, kernel, iterations=1)

    bitwise_and = cv2.bitwise_and(dilate_row, dilate_col)

    add = cv2.add(dilate_row, dilate_col)

    index_array = np.where(bitwise_and > 0)

    # print(np.where(dilate_col > 0))
    #
    # for index, i in enumerate(split_row_to_cell(split_to_row(index_array))):
    #     print(i)
    #     cv2.imwrite(f"/home/dong/tmp/save{index}.png", img[i[0][1]: i[2][1], i[0][0]: i[1][0]])

    cv2.imshow("dilate_row", dilate_row)
    cv2.imshow("dilate_col", dilate_col)
    cv2.imshow("bitwise_and", bitwise_and)
    cv2.imshow("img", img)

    cv2.waitKey()



if __name__ == "__main__":
    main()
