# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'prompt.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(378, 141)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(100, 90, 251, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.prompt = QtWidgets.QLabel(Dialog)
        self.prompt.setGeometry(QtCore.QRect(20, 20, 331, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        self.prompt.setFont(font)
        self.prompt.setObjectName("prompt")
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(70, 50, 231, 31))
        self.lineEdit.setMaxLength(5)
        self.lineEdit.setObjectName("lineEdit")
        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.prompt.setText(_translate("Dialog", "输入信息："))

