import cv2
import os
import numpy as np

# 参数调节 -- 默认
params_margin_x = 20  # x轴点相距达到一定的距离，才算两个有效点, 确定线的个数
params_margin_y = 20  # y轴点相距达到一定的距离，才算两个有效点， 确定线的个数
params_dot_margin = 3  # 和平均线的偏移量（对缝隙起作用，可去除点，也可变为独立一个点） 太大：被当成另一条直线上，太小：不把它当成一个独立的点
params_line_x = 10  # x上点个数的差值调节（线不均匀，有的粗有的细，甚至有的不连续）
params_line_y = 10  # y上点个数的差值调节（线不均匀，有的粗有的细，甚至有的不连续）


def recognize_line_x(line_xs, line_ys, num, num1, num2):
    y_line_list = []

    for k in [-3, -2, -1, 0, 1, 2, 3]:
        for i in range(len(line_xs)):
            if line_xs[i] == num + k:
                if line_ys[i] >= num1 and line_ys[i] <= num2 and line_ys[i] not in y_line_list:
                    y_line_list.append(line_ys[i])
    len_list = len(y_line_list)

    return len_list


def recognize_line_y(line_xs, line_ys, num, num1, num2):
    x_line_list = []
    for k in [-3, -2, -1, 0, 1, 2, 3]:
        for i in range(len(line_xs)):
            if line_ys[i] == num + k:
                if line_xs[i] >= num1 and line_xs[i] <= num2 and line_xs[i] not in x_line_list:
                    x_line_list.append(line_xs[i])
    len_list = len(x_line_list)

    return len_list


def split_image(name, save_path):
    image = cv2.imread(name)
    # 二值化(自适应阈值二值化)
    """
    dst = cv2.adaptiveThreshold(src, maxval, thresh_type, type, BlockSize, C)
    src： 输入图，只能输入单通道图像，通常来说为灰度图
    dst： 输出图
    maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
    thresh_type： 阈值的计算方法，包含以下2种类型：cv2.ADAPTIVE_THRESH_MEAN_C(通过平均方法取平均值)； cv2.ADAPTIVE_THRESH_GAUSSIAN_C（通过高斯）.
    type：二值化操作的类型，与固定阈值函数相同，包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV.
    BlockSize： 图片中分块的大小
    C ：阈值计算方法中的常数项
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
    cv2.imwrite('cell.jpg', binary)

    rows, cols = binary.shape
    print(f'{rows=}, {cols=}')

    # 原理：这个参数决定了 横线或者竖线的长度
    scale = 20  # 调节是否能精确识别点（粗、模糊 +   细、清晰  -）
    # 识别横线
    """
    腐蚀是一种消除边界点，使边界向内部收缩的过程   可以用来消除小且无意义的物体.
    膨胀是将与物体接触的所有背景点合并到该物体中，使边界向外部扩张的过程   可以用来填补物体中的空洞.

    用 cols // scale x 1 的 kernel，扫描图像的每一个像素；
    用 kernel 与其覆盖的二值图像做 “与” 操作；
    如果都为1，结果图像的该像素为1；否则为0.
    """
    # 矩形
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))  # 用长为 cols//scale ，宽为1 的矩形扫描，能够得到横线
    eroded = cv2.erode(binary, kernel, iterations=1)  # 腐蚀
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1)  # 膨胀
    cv2.imwrite('dilated1.jpg', dilatedcol)

    # 识别竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))  # 用长为1， 宽为rows//scale 的矩形扫描，能够得到竖线
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
    cv2.imwrite('dilated2.jpg', dilatedrow)

    # 标识交点
    bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)
    cv2.imwrite('bitwise.jpg', bitwiseAnd)

    # 标识表格
    merge = cv2.add(dilatedcol, dilatedrow)
    cv2.imwrite('add.jpg', merge)

    # 识别表格中所有的线
    line_ys, line_xs = np.where(merge > 0)

    # 识别黑白图中的白色点
    ys, xs = np.where(bitwiseAnd > 0)  # 交点附近的坐标

    mylisty = []
    mylistx = []

    print(ys, xs)

    # 通过排序，获取跳变的x和y的值，说明是交点，否则交点会有好多像素值，我只取最后一点
    i = 0
    myxs = np.sort(xs)
    # print('myxs', myxs)
    for i in range(len(myxs) - 1):
        if (myxs[i + 1] - myxs[i] > params_margin_x):
            mylistx.append(myxs[i])
        i = i + 1
    mylistx.append(myxs[i])  # 包括最后一点，每个点都取的是最大的一个
    print('纵向：', mylistx)
    print('纵向线数：', len(mylistx))

    i = 0
    myys = np.sort(ys)
    # print('myys', myys)

    for i in range(len(myys) - 1):
        if (myys[i + 1] - myys[i] > params_margin_y):
            mylisty.append(myys[i])
        i = i + 1
    mylisty.append(myys[i])
    print('横向', mylisty)
    print('横向线数：', len(mylisty))

    data_dict = {}
    data_list = []
    for i in range(len(myys)):
        for m in mylisty:
            for n in mylistx:
                if abs(m - ys[i]) < params_dot_margin and abs(n - xs[i]) < params_dot_margin and (m, n) not in data_list:
                    data_list.append((m, n))

        # print('@@@@@@@@@@@@', (ys[i], xs[i]))
    print('data_list', data_list)
    print(len(data_list))

    for m in range(len(mylisty)):
        line_list = []
        for i in data_list:
            if i[0] == mylisty[m]:
                line_list.append(i)
        data_dict[m] = sorted(line_list, key=lambda x: x[1])
    print('data_dict', data_dict)

    for i in range(len(data_dict) - 1):
        for index, value in enumerate(data_dict[i]):
            m = i
            if index == len(data_dict[i]) - 1:
                break

            for nn in range(1, len(data_dict[i])):
                m = i
                mark_num = 0
                n = index + nn
                if n == len(data_dict[i]):
                    break

                while m <= len(data_dict) - 2:  # recognize_line(line_xs, line_ys, 161, 57, 88)
                    if value[1] in [i[1] for i in data_dict[m + 1]] and data_dict[i][n][1] in [i[1] for i in data_dict[
                        m + 1]] and abs(
                            recognize_line_x(line_xs, line_ys, value[1], value[0], data_dict[m + 1][0][0]) - (
                                    data_dict[m + 1][0][0] - value[0])) <= params_line_y and abs(
                            recognize_line_x(line_xs, line_ys, data_dict[i][n][1], value[0], data_dict[m + 1][0][0]) - (
                                    data_dict[m + 1][0][0] - value[0])) <= params_line_y and abs(
                            recognize_line_y(line_xs, line_ys, value[0], value[1], data_dict[i][n][1]) - (
                                    data_dict[i][n][1] - value[1])) <= params_line_x and abs(
                            recognize_line_y(line_xs, line_ys, data_dict[m + 1][0][0], value[1], data_dict[i][n][1]) - (
                                    data_dict[i][n][1] - value[1])) <= params_line_x:
                        mark_num = 1
                        ROI = image[value[0]:data_dict[m + 1][0][0], value[1]:data_dict[i][n][1]]

                        order_num1 = mylisty.index(value[0])
                        order_num2 = mylisty.index(data_dict[m + 1][0][0]) - 1
                        order_num3 = mylistx.index(value[1])
                        order_num4 = mylistx.index(data_dict[i][n][1]) - 1

                        img_name = name.split('/')[1].split('.')[0] + '_' + str(order_num1) + '_' + str(
                            order_num2) + '_' + str(order_num3) + '_' + str(order_num4) + '.jpg'

                        save_path_detail = save_path + 'img_' + name.split('/')[1].split('.')[0]
                        if os.path.exists(save_path_detail):
                            pass
                        else:
                            os.mkdir(save_path_detail)
                        img_path = save_path_detail + '/' + img_name
                        cv2.imwrite(img_path, ROI)
                        break
                    else:
                        m += 1

                if mark_num == 1:
                    break


if __name__ == '__main__':
    # img_path = 'table_images/'
    # save_path = 'small_images/'
    # for i in os.listdir(img_path):
    #     if i == '12.png' or i == '13.png' or i == '23.png' or i == '7.png' or i == '15.png':
    #         pass
    #     else:
    #         split_image(img_path + i, save_path)

    split_image('/home/dong/tmp/tmp.jpg', 'small_images')