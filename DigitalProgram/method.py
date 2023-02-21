import os
import random
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QGraphicsPixmapItem, QGraphicsScene, \
    QMessageBox
import torch
from PIL import Image
from torchvision import transforms
from promptWindow import PromptWindow


class Method():
    # def __init__(self):
    #     self.findImg = False
    #     self.path = ""
    #     self.saveImg = None

    def find_img(self):
        filename, _ = QFileDialog.getOpenFileName(self.centralwidget, '选择图片', './data/',
                                                  'ALL(*.*);;Images(*.png *.jpg)')
        if filename:
            self.findImg = True
            self.path = filename
        return filename

    def open_img(self):
        filename = self.find_img()
        if filename:
            self.graphicsView.scene_img = QGraphicsScene()
            imgShow = QPixmap()
            imgShow.load(filename)
            imgShowItem = QGraphicsPixmapItem()
            imgShowItem.setPixmap(QPixmap(imgShow))
            self.graphicsView.scene_img.addItem(imgShowItem)
            self.graphicsView.setScene(self.graphicsView.scene_img)

    def show_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.shape[1]
        y = img.shape[0]
        frame = QImage(img, x, y, x * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView_2.setScene(scene)

        # self.graphicsView_2.scene_img = QGraphicsScene()
        # imgShow = QPixmap(img)
        # # imgShow.load(img)
        # imgShowItem = QGraphicsPixmapItem()
        # imgShowItem.setPixmap(imgShow)
        # self.graphicsView_2.scene_img.addItem(imgShowItem)
        # self.graphicsView_2.setScene(self.graphicsView_2.scene_img)

    def save_img(self):
        file_path = QFileDialog.getSaveFileName(self, '选择保存位置', 'C:/Users/Lance Song/Pictures/*.png',
                                                'Image files(*.png)')
        file_path = file_path[0]
        if file_path:
            print('file_path: ', file_path)
            cv2.imwrite(file_path, self.saveImg)

    def exit(self):
        res = QMessageBox.warning(self, "退出程序", "确定退出程序？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if res == QMessageBox.Yes:
            sys.exit()

    def gray(self, img=None):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            if img.ndim > 2:
                b = img[:, :, 0].copy()
                g = img[:, :, 1].copy()
                r = img[:, :, 2].copy()
                img = 0.2126 * r + 0.7152 * g + 0.0722 * b
            self.saveImg = img.astype(np.uint8)
            self.show_img(self.saveImg)
            return img

    def thresholdImg(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path.replace('\n', ''), cv2.IMREAD_UNCHANGED)
            if img.ndim == 3:
                img = self.gray(img)
            height, width = img.shape[0], img.shape[1]
            for h in range(height):
                for w in range(width):
                    if img[h, w] < 128:
                        img[h, w] = 0
                    else:
                        img[h, w] = 255
            self.saveImg = img.astype(np.uint8)
            self.show_img(self.saveImg)

    def reducecolor(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            height, width, channel = img.shape
            for c in range(channel):
                for h in range(height):
                    for w in range(width):
                        if img[h, w, c] >= 0 and img[h, w, c] < 64:
                            img[h, w, c] = 32
                        elif img[h, w, c] >= 64 and img[h, w, c] < 128:
                            img[h, w, c] = 96
                        elif img[h, w, c] >= 128 and img[h, w, c] < 192:
                            img[h, w, c] = 160
                        else:
                            img[h, w, c] = 224
            self.saveImg = img
            self.show_img(self.saveImg)

    def avgpooling(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            g = 8
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            resu = img.copy()
            H, W, C = img.shape
            Nh = int(H / g)
            Nw = int(W / g)
            for y in range(Nh):
                for x in range(Nw):
                    for c in range(C):
                        resu[g * y: g * (y + 1), g * x: g * (x + 1), c] = np.mean(
                            resu[g * y: g * (y + 1), g * x: g * (x + 1), c]).astype(np.int8)
            self.saveImg = resu
            self.show_img(self.saveImg)

    def maxpooling(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            G = 8
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            resu = img.copy()
            H, W, C = img.shape
            H = int(H / G)
            W = int(W / G)
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        resu[h * G: (h + 1) * G, w * G: (w + 1) * G, c] = np.max(
                            img[h * G: (h + 1) * G, w * G: (w + 1) * G, c])
            self.saveImg = resu
            self.show_img(self.saveImg)

    def cv2file(self, img):
        cv2.imwrite('./tmp.png', img)
        tmp = cv2.imread('./tmp.png')
        os.remove('./tmp.png')
        return tmp

    def bl_interpolate(self, img, ax=1.1, ay=1.1):
        """
        双线性插值
        :param ax:
        :param ay:
        :return:
        """
        # img = img[..., :min(3, img.shape[-1])]
        H, W, C = img.shape
        aH = int(ay * H)
        aW = int(ax * W)
        out = cv2.resize(img, (aW, aH), interpolation=cv2.INTER_LINEAR)
        return out

    def magnify(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            promWin = PromptWindow(parent=self, window_title='图片缩放', prompt_text='请输入缩放倍率: ')
            promWin.show()
            ret = promWin.exec()
            if ret == 1 and promWin.result is not None:
                num = float(promWin.result)
                img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
                out = self.bl_interpolate(img, num, num)
                self.saveImg = self.cv2file(out)
                self.show_img(self.saveImg)

    def suofangImg(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            value = self.suofang.value() * 1.0 / 10
            print('scale value: ', value)
            img = self.bl_interpolate(img.copy(), value, value)
            self.saveImg = self.cv2file(img)
            self.show_img(self.saveImg)

    def lightImg(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            value = self.light.value()
            value = value * 1.0 / 10
            print(f'亮度：{value}')
            rows, cols, channels = img.shape
            blank = np.zeros([rows, cols, channels], img.dtype)
            res = cv2.addWeighted(img, value, blank, 1, 3)
            self.saveImg = res
            self.show_img(self.saveImg)

    def xuanzhuanImg(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            # 旋转中心坐标，逆时针旋转：-90°，缩放因子：1
            M_2 = cv2.getRotationMatrix2D(center, self.xuanzhuan.value(), 1)
            img = cv2.warpAffine(img, M_2, (w, h))
            # cv2.imshow("./rotated_-90.jpg", rotated_2)
            self.saveImg = img
            self.show_img(self.saveImg)

    # 噪声
    def gaussian(self):
        '''
                    添加高斯噪声
                    mean : 均值
                    var : 方差
        '''
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            mean = 0
            var = 0.01
            image = np.array(img / 255, dtype=float)
            noise = np.random.normal(mean, var ** 0.5, image.shape)
            out = image + noise
            if out.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.
            out = np.clip(out, low_clip, 1.0)
            out = np.uint8(out * 255)
            self.saveImg = out
            self.show_img(self.saveImg)

    def spicesalt(self):
        '''
            添加椒盐噪声
            prob:噪声比例
        '''
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)

        else:
            prob = 0.05
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            image = img.copy()
            output = np.zeros(image.shape, np.uint8)
            thres = 1 - prob
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    rdn = random.random()
                    if rdn < prob:
                        output[i][j] = 0
                    elif rdn > thres:
                        output[i][j] = 255
                    else:
                        output[i][j] = image[i][j]
            self.saveImg = output
            self.show_img(self.saveImg)


    def poisson(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            img = img.copy()
            img = img.astype(float)
            noise_mask = np.random.poisson(img)
            noisy_img = img + noise_mask
            noisy_img = noisy_img.astype(np.uint8)
            self.saveImg = noisy_img
            self.show_img(self.saveImg)


    def spot(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            mean = 0.0
            var = 0.2
            copyImage = img.copy()
            noise = np.random.normal(loc=mean, scale=var, size=copyImage.shape)
            copyImage = np.array(copyImage / 255, dtype=float)
            out = (1 + noise) * copyImage
            out = np.clip(out, 0.0, 1.0)
            out = np.uint8(out * 255)
            self.saveImg = out
            self.show_img(self.saveImg)


    def fourier(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            img = np.mean(img[..., :min(3, img.shape[-1])], axis=2)
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)  # 得到结果为复数
            magnitude_spectrum = 10 * np.log(np.abs(fshift))  # 先取绝对值，表示取模。取对数，将数据范围变小
            self.saveImg = magnitude_spectrum.astype(np.uint8)
            print(magnitude_spectrum.shape)
            print(magnitude_spectrum.ndim)
            self.show_img(self.saveImg)


    def disperse(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            img = np.mean(img[..., :min(3, img.shape[-1])], axis=2)
            print('img.shape: ', img.shape)
            h, w = img.shape[:3]
            img = img[:(h // 2 * 2), :(w // 2 * 2)]
            img = img.astype(np.float)
            # 进行离散余弦变换
            img_dct = cv2.dct(img)
            # 进行log处理
            img_dct_log = np.log(abs(img_dct))
            img_dct_log = img_dct_log * 32
            self.saveImg = img_dct_log.astype(np.uint8)
            self.show_img(self.saveImg)


    def gaussiansmoothing(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            K_size = 3
            sigma = 1.3
            if len(img.shape) == 3:
                H, W, C = img.shape
            else:
                img = np.expand_dims(img, axis=-1)
                H, W, C = img.shape
            # zero padding
            pad = K_size // 2
            resu = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float32)
            resu[pad:pad + H, pad:pad + W] = img.copy().astype(np.float32)
            # Kernel
            K = np.zeros((K_size, K_size), dtype=np.float32)
            for x in range(-pad, -pad + K_size):
                for y in range(-pad, -pad + K_size):
                    K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
            K /= (2 * np.pi * sigma * sigma)
            K /= K.sum()
            tmp = resu.copy()
            # filtering
            for y in range(H):
                for x in range(W):
                    for c in range(C):
                        resu[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
            resu = np.clip(resu, 0, 255)
            resu = resu[pad:pad + H, pad:pad + W].astype(np.uint8)
            self.saveImg = resu
            self.show_img(self.saveImg)


    def midvalue(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            K_size = 3
            H, W, C = img.shape
            # zeropadding
            pad = K_size // 2
            resu = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float32)
            resu[pad:pad + H, pad:pad + W] = img.copy().astype(np.float32)
            tmp = resu.copy()
            # filtering
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        resu[pad + h, pad + w, c] = np.median(tmp[h:h + K_size, w:w + K_size, c])
            resu = resu[pad:pad + H, pad:pad + W].astype(np.uint8)
            self.saveImg = resu
            self.show_img(self.saveImg)


    def meanvalue(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            K_size = 3
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            H, W, C = img.shape
            pad = K_size // 2
            resu = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float32)
            resu[pad: pad + H, pad: pad + W] = img.copy().astype(np.float32)
            tmp = resu.copy()
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        resu[pad + h, pad + w, c] = np.mean(tmp[h: h + K_size, w: w + K_size, c])
            resu = resu[pad: pad + H, pad: pad + W].astype(np.uint8)
            self.saveImg = resu
            self.show_img(self.saveImg)


    def ruihua(self):  # 线性
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            kernel = np.float32([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            out = cv2.filter2D(img, -1, kernel)
            self.saveImg = out.astype(np.uint8)
            self.show_img(self.saveImg)


    def ruihua1(self):  # 非线性
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_ANYCOLOR)
            kernel = np.float32([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            out = cv2.filter2D(img, -1, kernel)
            self.saveImg = out.astype(np.uint8)
            self.show_img(self.saveImg)


    def Bhist(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            plt.hist(img[:, :, 0].flatten(), 256)
            plt.title('B hist')
            plt.savefig('./tmp.jpg', bbox_inches='tight', dpi=70)
            plt.close()
            res = cv2.imread('./tmp.jpg')
            os.remove('./tmp.jpg')
            self.saveImg = res.astype(np.uint8)
            self.show_img(self.saveImg)


    def Ghist(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            plt.hist(img[:, :, 1].flatten(), 256)
            plt.title('G hist')
            plt.savefig('./tmp.jpg', bbox_inches='tight', dpi=70)
            plt.close()
            res = cv2.imread('./tmp.jpg')
            os.remove('./tmp.jpg')
            self.saveImg = res.astype(np.uint8)
            self.show_img(self.saveImg)


    def Rhist(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            plt.hist(img[:, :, 2].flatten(), 256)
            plt.title('R hist')
            plt.savefig('./tmp.jpg', bbox_inches='tight', dpi=70)
            plt.close()
            res = cv2.imread('./tmp.jpg')
            os.remove('./tmp.jpg')
            self.saveImg = res.astype(np.uint8)
            self.show_img(self.saveImg)


    def gamma_enhance(self, mat, gamma=0.9):
        tar = mat.copy()
        tar = tar * 1.0 / 255
        tar = np.power(tar, gamma)
        tar = (tar * 255).astype(np.uint8)
        return tar


    def HSV(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            gamma = 1.8
            res = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            res[:, :, 1] = self.gamma_enhance(res[:, :, 1], gamma)
            res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
            self.saveImg = res
            self.show_img(self.saveImg)


    def YCbcr(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            gamma = 0.8
            res = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            res[:, :, 0] = self.gamma_enhance(res[:, :, 0], gamma)
            res = cv2.cvtColor(res, cv2.COLOR_YCR_CB2BGR)
            self.saveImg = res
            self.show_img(self.saveImg)


    def histmean(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            # 通道分解
            b = img[:, :, 0]
            g = img[:, :, 1]
            r = img[:, :, 2]
            bH = cv2.equalizeHist(b)
            gH = cv2.equalizeHist(g)
            rH = cv2.equalizeHist(r)
            # 通道合成
            img[:, :, 0] = bH
            img[:, :, 1] = gH
            img[:, :, 2] = rH
            self.saveImg = img.astype(np.uint8)
            self.show_img(self.saveImg)


    # 腐蚀
    def corrosion(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            # 结构元素
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # 腐蚀图像
            res = cv2.erode(img, kernel)
            self.saveImg = res
            self.show_img(self.saveImg)


    # 膨胀
    def expand(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            # 结构元素
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # 膨胀图像
            res = cv2.dilate(img, kernel)
            self.saveImg = res
            self.show_img(self.saveImg)


    # 开运算
    def openoperation(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            self.saveImg = res
            self.show_img(self.saveImg)


    # 闭运算
    def closeoperation(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            res = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            self.saveImg = res
            self.show_img(self.saveImg)


    def hood(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            H, W, C = img.shape
            # # Otsu binary
            # # Grayscale
            out = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
            out = out.astype(np.uint8)
            # Determine threshold of Otsu's binarization
            max_sigma = 0
            max_t = 0
            for _t in range(1, 255):
                v0 = out[np.where(out < _t)]
                m0 = np.mean(v0) if len(v0) > 0 else 0.
                w0 = len(v0) / (H * W)
                v1 = out[np.where(out >= _t)]
                m1 = np.mean(v1) if len(v1) > 0 else 0.
                w1 = len(v1) / (H * W)
                sigma = w0 * w1 * ((m0 - m1) ** 2)
                if sigma > max_sigma:
                    max_sigma = sigma
                    max_t = _t
            # Binarization
            th = max_t
            out[out < th] = 0
            out[out >= th] = 255
            # 设置卷积核
            kernel = np.ones((3, 3), np.uint8)
            # 顶帽运算
            dst = cv2.morphologyEx(out, cv2.MORPH_TOPHAT, kernel)
            self.saveImg = dst
            self.show_img(self.saveImg)


    def black(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            H, W, C = img.shape
            # Otsu binary
            # Grayscale
            out = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
            out = out.astype(np.uint8)
            # Determine threshold of Otsu's binarization
            max_sigma = 0
            max_t = 0
            for _t in range(1, 255):
                v0 = out[np.where(out < _t)]
                m0 = np.mean(v0) if len(v0) > 0 else 0.
                w0 = len(v0) / (H * W)
                v1 = out[np.where(out >= _t)]
                m1 = np.mean(v1) if len(v1) > 0 else 0.
                w1 = len(v1) / (H * W)
                sigma = w0 * w1 * ((m0 - m1) ** 2)
                if sigma > max_sigma:
                    max_sigma = sigma
                    max_t = _t
            # Binarization
            th = max_t
            out[out < th] = 0
            out[out >= th] = 255
            # 设置卷积核
            kernel = np.ones((3, 3), np.uint8)
            # 黑帽运算
            dst = cv2.morphologyEx(out, cv2.MORPH_BLACKHAT, kernel)
            self.saveImg = dst
            self.show_img(self.saveImg)
            return self.saveImg


    def morphology(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            kernel = np.ones((3, 3), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
            self.saveImg = img
            self.show_img(self.saveImg)


    def thresholdsegmentation(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            promWin = PromptWindow(parent=self, window_title='阈值分割', prompt_text='请输入0~255的整数: ')
            promWin.show()
            ret = promWin.exec()
            if ret == 1 and promWin.result is not None:
                thred = int(promWin.result)
                img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
                img = self.gray(img)
                _, out = cv2.threshold(img, thred, 255, cv2.THRESH_BINARY)
                self.saveImg = out.astype(np.uint8)
                self.show_img(self.saveImg)


    def featureExtract_img(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            # kaze检测
            kaze = cv2.KAZE_create()
            keypoints = kaze.detect(img, None)
            img1 = img.copy()
            kaze_img = cv2.drawKeypoints(img, keypoints, img1, color=(0, 255, 0))
            self.saveImg = kaze_img
            self.show_img(self.saveImg)


    def imgclassification(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            label_map = {
                0: "cat",
                1: "dog",
            }
            data_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torch.load("./model/AlexNet.pth")
            model.eval()
            model.to(device)
            img = Image.open(self.path)
            img_data = data_transform(img)
            img_data = torch.unsqueeze(img_data, dim=0)
            predict = model(img_data.to(device))
            predict = torch.argmax(predict).cpu().numpy()
            if predict == 0:
                QMessageBox.about(self, "预测", "预测为猫类!")
            else:
                QMessageBox.about(self, "预测", "预测为狗类!")


    def startprogram(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            self.saveImg = img
            self.show_img(img)


    def removeImg(self):
        res = QMessageBox.warning(self, "移除确认", "是否确认移除当前图片", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if res == QMessageBox.Yes:
            self.saveImg = None
            scene = QGraphicsScene()
            self.graphicsView.setScene(scene)
