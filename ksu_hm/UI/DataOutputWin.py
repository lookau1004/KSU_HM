# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DataOutputWin.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_OutputWindow(object):
    def setupUi(self, OutputWindow):
        OutputWindow.setObjectName("OutputWindow")
        OutputWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(OutputWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.WinOutputList = QtWidgets.QListWidget(self.centralwidget)
        self.WinOutputList.setGeometry(QtCore.QRect(10, 10, 771, 551))
        self.WinOutputList.setObjectName("WinOutputList")
        OutputWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(OutputWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        OutputWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(OutputWindow)
        self.statusbar.setObjectName("statusbar")
        OutputWindow.setStatusBar(self.statusbar)

        self.retranslateUi(OutputWindow)
        QtCore.QMetaObject.connectSlotsByName(OutputWindow)

    def retranslateUi(self, OutputWindow):
        _translate = QtCore.QCoreApplication.translate
        OutputWindow.setWindowTitle(_translate("OutputWindow", "Output Window"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    OutputWindow = QtWidgets.QMainWindow()
    ui = Ui_OutputWindow()
    ui.setupUi(OutputWindow)
    OutputWindow.show()
    sys.exit(app.exec_())
