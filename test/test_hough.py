import cv2
import numpy as np
import pathlib


def degree(theta):
    ret = theta / np.pi * 180
    return min(abs(90 - ret), ret)


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def img_degree(img):
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 800)

    if lines is None: return 0

    degree_list = []
    for line in lines:
        rho, theta = line[0]
        degree_list.append(degree(theta))
    return np.min(degree_list)


def main():

    for path in pathlib.Path("/home/dong/tmp/zuowen/img/0").glob("**/*.jpg"):
        print("*"*100)
        img = cv2.imread(str(path))

        edges = cv2.Canny(img, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 800)

        if lines is not None:
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
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                print(f'{rho=},{theta=}', {degree(theta)})

        degree_xx = img_degree(img)

        img = rotate(img, degree_xx)
        img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
        img = cv2.putText(img, str(degree_xx), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
        cv2.imshow("", img)
        cv2.waitKey(0)
        cv2.imwrite("/home/dong/tmp/tmp.jpg", img)


if __name__ == "__main__":
    main()
