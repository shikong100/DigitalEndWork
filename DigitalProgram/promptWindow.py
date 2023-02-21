from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QDialog

from ui.prompt import Ui_Dialog
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRegExp

import sys


class PromptWindow(QDialog, Ui_Dialog):
    def __init__(self, parent=None, window_title = '请输入', prompt_text='输入：'):
        super(PromptWindow, self).__init__(parent)
        self.setupUi(self)

        self.setWindowTitle(window_title)
        self.prompt.setText(prompt_text)

        reg = QRegExp('[.0-9]+$')
        validator = QRegExpValidator(self)
        validator.setRegExp(reg)
        self.lineEdit.setValidator(validator)

        self.buttonBox.accepted.connect(self.prog)
        self.buttonBox.rejected.connect(self.canceled)

        self.result = None

    def prog(self):
        value = self.lineEdit.text()
        if value.replace('.', '') == '':
            self.result = None
        else:
            self.result = float(value)

        return self.result

    def canceled(self):
        self.result = None
        print('canceled!')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    mainWindon = PromptWindow()
    mainWindon.show()

    sys.exit(app.exec_())

