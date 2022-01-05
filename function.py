import cv2
import numpy as np

import os
import json
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 10000


# 读取图片文件
def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


# 图像去噪灰度处理
def gray_guss(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image



class CardPredictor:
    def __init__(self):
        # 车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
        f = open('config.js')
        j = json.load(f)
        for c in j["config"]:
            if c["open"]:
                self.cfg = c.copy()
                break
        else:
            raise RuntimeError('没有设置有效配置参数')



    def accurate_place(self, card_img_hsv, limit1, limit2, color):
        row_num, col_num = card_img_hsv.shape[:2]
        xl = col_num
        xr = 0
        yh = 0
        yl = row_num
        # col_num_limit = self.cfg["col_num_limit"]
        row_num_limit = self.cfg["row_num_limit"]
        col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
        for i in range(row_num):
            count = 0
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > col_num_limit:
                if yl > i:
                    yl = i
                if yh < i:
                    yh = i
        for j in range(col_num):
            count = 0
            for i in range(row_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > row_num - row_num_limit:
                if xl > j:
                    xl = j
                if xr < j:
                    xr = j
        return xl, xr, yh, yl

    def predict(self, car_pic):
        if type(car_pic) == type(""):
            img = imreadex(car_pic)
        else:
            img = car_pic
        pic_hight, pic_width = img.shape[:2]

        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)

        blur = self.cfg["blur"]
        # 高斯去噪
        if blur > 0:
            img = cv2.GaussianBlur(img, (blur, blur), 0)  # 图片分辨率调整
        oldimg = img
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #灰度处理
        Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)   #sobel算子边缘检测
        absX = cv2.convertScaleAbs(Sobel_x)   #转回uint8
        image = absX
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)  #自适应阈值处理
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 5))  #闭运算，白色部分练成整体
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX, iterations=1)
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))  #去除小白点
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
        image = cv2.dilate(image, kernelX)  #膨胀
        image = cv2.erode(image, kernelX)   #腐蚀
        image = cv2.erode(image, kernelY)   #腐蚀
        image = cv2.dilate(image, kernelY)  #膨胀
        image = cv2.medianBlur(image, 15)  #中值滤波去除噪点
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#轮廓检测
        # 一一排除不是车牌的矩形区域
        car_contours = []  #筛选车牌位置的轮廓
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            area_width, area_height = rect[1]
            if area_width < area_height:
                area_width, area_height = area_height, area_width
            wh_ratio = area_width / area_height
            # print(wh_ratio)
            # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
            if wh_ratio > 2 and wh_ratio < 5.5:
                car_contours.append(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

        print("精确定位")
        card_imgs = []
        # 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
        for rect in car_contours:
            if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
                angle = 1
            else:
                angle = rect[2]
            rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除

            box = cv2.boxPoints(rect)
            heigth_point = right_point = [0, 0]
            left_point = low_point = [pic_width, pic_hight]
            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if low_point[1] > point[1]:
                    low_point = point
                if heigth_point[1] < point[1]:
                    heigth_point = point
                if right_point[0] < point[0]:
                    right_point = point

            if left_point[1] <= right_point[1]:  # 正角度
                new_right_point = [right_point[0], heigth_point[1]]
                pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, heigth_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                point_limit(new_right_point)
                point_limit(heigth_point)
                point_limit(left_point)
                card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
                card_imgs.append(card_img)
            elif left_point[1] > right_point[1]:  # 负角度

                new_left_point = [left_point[0], heigth_point[1]]
                pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, heigth_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                point_limit(right_point)
                point_limit(heigth_point)
                point_limit(new_left_point)
                card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
                card_imgs.append(card_img)

            #cv2.waitKey(0)
        # 开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
        colors = []
        for card_index, card_img in enumerate(card_imgs):
                    green = yello = blue = black = white = 0
                    card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
                    # 有转换失败的可能，原因来自于上面矫正矩形出错
                    if card_img_hsv is None:
                        continue
                    row_num, col_num = card_img_hsv.shape[:2]
                    card_img_count = row_num * col_num

                    for i in range(row_num):
                        for j in range(col_num):
                            H = card_img_hsv.item(i, j, 0)
                            S = card_img_hsv.item(i, j, 1)
                            V = card_img_hsv.item(i, j, 2)
                            if 11 < H <= 34 and S > 34:  # 图片分辨率调整
                                yello += 1
                            elif 35 < H <= 99 and S > 34:  # 图片分辨率调整
                                green += 1
                            elif 99 < H <= 124 and S > 34:  # 图片分辨率调整
                                blue += 1

                            if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                                black += 1
                            elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                                white += 1
                    color = "no"

                    limit1 = limit2 = 0
                    if yello * 2 >= card_img_count:
                        color = "yello"
                        limit1 = 11
                        limit2 = 34  # 有的图片有色偏偏绿
                    elif green * 2 >= card_img_count:
                        color = "green"
                        limit1 = 35
                        limit2 = 99
                    elif blue * 2 >= card_img_count:
                        color = "blue"
                        limit1 = 100
                        limit2 = 124  # 有的图片有色偏偏紫
                    elif black + white >= card_img_count * 0.7:  # TODO
                        color = "bw"
                    print(color)
                    colors.append(color)
                    print(blue, green, yello, black, white, card_img_count)

                    if limit1 == 0:
                        continue
                    # 以上为确定车牌颜色
                    # 以下为根据车牌颜色再定位，缩小边缘非车牌边界
                    xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
                    if yl == yh and xl == xr:
                            continue
                    need_accurate = False
                    if yl >= yh:
                            yl = 0
                            yh = row_num
                            need_accurate = True
                    if xl >= xr:
                            xl = 0
                            xr = col_num
                            need_accurate = True
                    card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (
                                    yh - yl) // 4 else card_img[
                                                       yl - (
                                                               yh - yl) // 4:yh,
                                                       xl:xr]
                    if need_accurate:  # 可能x或y方向未缩小，需要再试一次
                            card_img = card_imgs[card_index]
                            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
                            xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
                            if yl == yh and xl == xr:
                                continue
                            if yl >= yh:
                                yl = 0
                                yh = row_num
                            if xl >= xr:
                                xl = 0
                                xr = col_num
                    card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (
                                    yh - yl) // 4 else card_img[
                                                       yl - (
                                                               yh - yl) // 4:yh,
                                                       xl:xr]

                    predict_result = []
                    word_images = []
                    roi = None
                    card_color = None
                    for i, color in enumerate(colors):
                        if color in ("blue", "yello", "green"):
                            card_img = card_imgs[i]  # 定位的车牌
                            gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                            # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反转
                            if color == "green" or color == "yello":
                                gray_img = cv2.bitwise_not(gray_img)
                            ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


                            ret, binary = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY_INV)  # 二值化
                            cv2.imshow('bin', binary)  # 显示二值过后的结果， 白底黑字
                            cv2.waitKey(0)
                            bin1 = cv2.resize(binary, (370, 82))  # 改变一下大小，有助于识别
                            kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))  # 获取一个卷积核，
                            dilated = cv2.dilate(bin1, kernel1)  # 白色区域膨胀
                            predict_result = pytesseract.image_to_string(dilated, lang='chi_sim')  # 识别
                            roi = card_img
                            card_color = color
                            print(predict_result)
                            break #插入测试

                    return predict_result, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色
