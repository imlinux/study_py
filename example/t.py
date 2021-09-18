import cv2


def data():
    with open("/home/dong/tmp/mnist/train.txt") as f:
        for i in f:
            yield i.replace("\n", "").split(" ")

img = cv2.imread("/home/dong/tmp/table.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)

rows, cols = binary.shape
scale = 20
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
eroded = cv2.erode(binary, kernel, iterations=1)
dilatedcol = cv2.dilate(eroded, kernel, iterations=1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
eroded = cv2.erode(binary, kernel, iterations=1)
dilatedrow = cv2.dilate(eroded, kernel, iterations=1)

bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)
merge = cv2.add(dilatedcol, dilatedrow)

cv2.imshow("", kernel)
cv2.waitKey(0)