import cv2
import numpy as np


def degree(theta):
    ret = theta / np.pi * 180

    if ret > 90:
        return ret - 90
    return round(ret, 0)


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def main():
    img = cv2.imread("/home/dong/tmp/SHENJIANG_F_00028.pdf/0.jpg", cv2.IMREAD_GRAYSCALE)

    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 360)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        print(f'{rho=},{theta=}', {degree(theta)})

    #img = rotate(img, 2)
    cv2.imshow("", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
