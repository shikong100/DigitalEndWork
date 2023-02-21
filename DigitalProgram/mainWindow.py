import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication

from ui.ui_mainWindow import Ui_mainWindow
from method import Method

from PyQt5 import QtCore


class Window(QMainWindow, Ui_mainWindow, Method):
    def __init__(self):
        # 从文件中加载UI定义
        super().__init__()
        self.setupUi(self)

        # 初始化成员变量
        self.findImg = False
        self.path = ""
        self.saveImg = None

        # 禁用窗口大小变换
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint)
        self.setFixedSize(self.width(), self.height())
        # 设置图标
        icon = QIcon('./Icon.png')
        self.setWindowIcon(icon)

        # 方法绑定
        # 文件
        self.action_open.triggered.connect(self.open_img)
        self.action_save.triggered.connect(self.save_img)
        self.action_exit.triggered.connect(self.exit)
        # 编辑
        self.action_gray.triggered.connect(self.gray)
        self.action_threshold.triggered.connect(self.thresholdImg)
        self.action_reducecolor.triggered.connect(self.reducecolor)
        self.action_avgpooling.triggered.connect(self.avgpooling)
        self.action_maxpooling.triggered.connect(self.maxpooling)
        self.action_magnify.triggered.connect(self.magnify)
        # 噪声
        self.action_gaussian.triggered.connect(self.gaussian)
        self.action_spicedsalt.triggered.connect(self.spicesalt)
        self.action_poisson.triggered.connect(self.poisson)
        self.action_spot.triggered.connect(self.spot)
        # 变换
        self.action_fourier.triggered.connect(self.fourier)
        self.action_disperse.triggered.connect(self.disperse)
        # 滤波
        self.action_gaussiansmoothing.triggered.connect(self.gaussiansmoothing)
        self.action_midvalue.triggered.connect(self.midvalue)
        self.action_meanvalue.triggered.connect(self.meanvalue)
        self.action_ruihua.triggered.connect(self.ruihua)
        self.action_ruihua1.triggered.connect(self.ruihua1)
        # 直方图
        self.action_B.triggered.connect(self.Bhist)
        self.action_G.triggered.connect(self.Ghist)
        self.action_R.triggered.connect(self.Rhist)
        # 图像增强
        self.action_HSV.triggered.connect(self.HSV)
        self.action_YCbcr.triggered.connect(self.YCbcr)
        self.action_histmean.triggered.connect(self.histmean)
        # 形态学操作
        self.action_corrosion.triggered.connect(self.corrosion)
        self.action_expand.triggered.connect(self.expand)
        self.action_openoperation.triggered.connect(self.openoperation)
        self.action_closeoperation.triggered.connect(self.closeoperation)
        self.action_hood.triggered.connect(self.hood)
        self.action_black.triggered.connect(self.black)
        self.action_morphology.triggered.connect(self.morphology)
        # 其他
        self.action_thresholdsegmentation.triggered.connect(self.thresholdsegmentation)
        self.actiont_featureExtract.triggered.connect(self.featureExtract_img)
        self.action_imgclassfication.triggered.connect(self.imgclassification)

        self.startProgramButton.clicked.connect(self.startprogram)
        self.exitProgram.clicked.connect(self.exit)
        self.browseButton.clicked.connect(self.open_img)
        self.saveButton.clicked.connect(self.save_img)
        self.removeButton.clicked.connect(self.removeImg)
        self.suofang.valueChanged.connect(self.suofangImg)
        self.light.valueChanged.connect(self.lightImg)
        self.xuanzhuan.valueChanged.connect(self.xuanzhuanImg)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())
