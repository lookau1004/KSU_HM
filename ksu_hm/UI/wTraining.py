# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'wTraining.UI'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1136, 757)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.WinCamaraLabel = QtWidgets.QLabel(self.centralwidget)
        self.WinCamaraLabel.setGeometry(QtCore.QRect(30, 30, 600, 400))
        self.WinCamaraLabel.setObjectName("WinCamaraLabel")
        self.WinStartBtn = QtWidgets.QPushButton(self.centralwidget)
        self.WinStartBtn.setGeometry(QtCore.QRect(930, 30, 75, 23))
        self.WinStartBtn.setObjectName("WinStartBtn")
        self.WinStopBtn = QtWidgets.QPushButton(self.centralwidget)
        self.WinStopBtn.setGeometry(QtCore.QRect(930, 70, 75, 23))
        self.WinStopBtn.setObjectName("WinStopBtn")
        self.WinExitBtn = QtWidgets.QPushButton(self.centralwidget)
        self.WinExitBtn.setGeometry(QtCore.QRect(930, 110, 75, 23))
        self.WinExitBtn.setObjectName("WinExitBtn")
        self.WinCaptureMotionBtn = QtWidgets.QPushButton(self.centralwidget)
        self.WinCaptureMotionBtn.setEnabled(False)
        self.WinCaptureMotionBtn.setGeometry(QtCore.QRect(900, 150, 141, 41))
        self.WinCaptureMotionBtn.setObjectName("WinCaptureMotionBtn")
        self.WinIndexLabelText = QtWidgets.QLabel(self.centralwidget)
        self.WinIndexLabelText.setGeometry(QtCore.QRect(950, 220, 54, 12))
        self.WinIndexLabelText.setObjectName("WinIndexLabelText")
        self.WinIndexLineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.WinIndexLineEdit.setGeometry(QtCore.QRect(910, 240, 113, 20))
        self.WinIndexLineEdit.setObjectName("WinIndexLineEdit")
        self.WinDataListWidget = QtWidgets.QListWidget(self.centralwidget)
        self.WinDataListWidget.setGeometry(QtCore.QRect(840, 290, 256, 401))
        self.WinDataListWidget.setObjectName("WinDataListWidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1136, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Gesture Training"))
        self.WinCamaraLabel.setText(_translate("MainWindow", "TextLabel"))
        self.WinStartBtn.setText(_translate("MainWindow", "Start"))
        self.WinStopBtn.setText(_translate("MainWindow", "Stop"))
        self.WinExitBtn.setText(_translate("MainWindow", "Exit"))
        self.WinCaptureMotionBtn.setText(_translate("MainWindow", "CaptureMotion"))
        self.WinIndexLabelText.setText(_translate("MainWindow", "Index"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
