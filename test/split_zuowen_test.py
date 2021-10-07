import cv2
import pathlib
import os
from pyzbar.pyzbar import decode
import numpy as np
from imutils import perspective
import imutils

from numpy import ones,vstack
from numpy.linalg import lstsq


def line(points):

    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
    return m, c


def rotate(image, center=None, scale=1.0):

    decodes = decode(image)

    if len(decodes) == 0: return 0, None

    d = decodes[0]
    polygon = d.polygon

    if polygon[1].x - polygon[0].x == 0: return 0, None

    minx = min(polygon[0].x, polygon[1].x, polygon[2].x, polygon[3].x)
    miny = min(polygon[0].y, polygon[1].y, polygon[2].y, polygon[3].y)

    slope = (polygon[1].y - polygon[0].y) / (polygon[1].x - polygon[0].x)
    rad = np.arctan(slope)

    deg = np.rad2deg(rad)

    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    deg = min(abs(90 - deg), deg)
    print(f"{rad=} {deg=} {polygon=}")
    M = cv2.getRotationMatrix2D(center, abs(deg), scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated, minx, miny


def split(img_path, output):
    *_, filename = img_path.split("/")
    img = cv2.imread(img_path)
    img, *polygon = rotate(img)

    decodes = decode(img)
    print(decodes[0].polygon)

    if polygon is None: return

    polygon = [1978, 3076]

    os.makedirs(output, exist_ok=True)

    row = polygon[1] - int(round(2268 - 2268 * 0.017542060057402487 , 0))
    col = polygon[0] - int(round(260 * (1 - 0.017542060057402487), 0))
    w = [0, 105, 207, 309, 411, 513, 615]
    h = 76

    for i in range(6):
        cv2.imwrite(f"{output}/{filename}-{row}-{i}.jpg", img[row: row + h, col + w[i]: col + w[i + 1]])

    row = polygon[1] + 168
    col = polygon[0] - 1755
    w = [0, 177, 357, 546, 732, 924, 1122, 1284, 1452]
    h = 120

    # for i in range(8):
    #     cv2.imwrite(f"{output}/{filename}-{row}-{i}.jpg", img[row: row + h, col + w[i] + 2: col + w[i + 1] - 2])


def main():
    img = cv2.imread("/home/dong/tmp/zuowen/img/0/JUYE_F_00019.pdf-81.jpg")
    polygon = decode(img)[0].polygon
    rect = perspective.order_points(np.array([(polygon[0].x, polygon[0].y), (polygon[1].x, polygon[1].y), (polygon[2].x, polygon[2].y), (polygon[3].x, polygon[3].y)]))
    (tl, tr, br, bl) = rect

    (row, col) = img.shape[:2]

    m1, c1 = line([tr, br])
    m2, c2 = line([bl, br])

    pts = np.array([(0, 0), (abs(c1//m1), 0), (0, abs(c2)), br])
    img = imutils.perspective.four_point_transform(img, pts)
    cv2.imwrite("/home/dong/tmp/tmp.jpg", img)

# split("/home/dong/tmp/zuowen/img/0/JUYE_F_00019.pdf-3.jpg", "/home/dong/tmp/tmp1")
# for f in pathlib.Path("/home/dong/tmp/zuowen/img/0").iterdir():
#     print(f)
#     split(str(f), "/home/dong/tmp/tmp1")

main()
