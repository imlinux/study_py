import cv2
import numpy as np


class SealRemove(object):
    """
    印章处理类
    """

    def remove_red(self, image):

        blue_c, green_c, red_c = cv2.split(image)

        red_part = red_c[-400:]

        thresh, ret = cv2.threshold(red_part, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        filter_condition = int(thresh * 0.95)
        _, red_thresh = cv2.threshold(red_part, filter_condition, 255, cv2.THRESH_BINARY)

        red_c[-400:] = red_thresh
        blue_c[-400:] = red_thresh
        green_c[-400:] = red_thresh

        result_blue = np.expand_dims(blue_c, axis=2)
        result_greep = np.expand_dims(green_c, axis=2)
        result_red = np.expand_dims(red_c, axis=2)
        result_img = np.concatenate((result_blue, result_greep, result_red), axis=-1)

        return result_img


if __name__ == '__main__':
    image = '/home/dong/tmp/1.jpg'
    img = cv2.imread(image)
    print(img.shape)
    seal_rm = SealRemove()
    rm_img = seal_rm.remove_red(img)
    cv2.imwrite("t.jpg", rm_img)