# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Timer = QtWidgets.QLabel(self.centralwidget)
        self.Timer.setGeometry(QtCore.QRect(30, 20, 61, 16))
        self.Timer.setObjectName("Timer")
        self.setTimer = QtWidgets.QTextEdit(self.centralwidget)
        self.setTimer.setGeometry(QtCore.QRect(100, 10, 111, 31))
        self.setTimer.setObjectName("setTimer")
        self.Apply = QtWidgets.QPushButton(self.centralwidget)
        self.Apply.setGeometry(QtCore.QRect(250, 20, 75, 23))
        self.Apply.setObjectName("Apply")
        self.CurrentTimer = QtWidgets.QLabel(self.centralwidget)
        self.CurrentTimer.setGeometry(QtCore.QRect(30, 60, 151, 31))
        self.CurrentTimer.setObjectName("CurrentTimer")
        self.cTimerValue = QtWidgets.QLabel(self.centralwidget)
        self.cTimerValue.setGeometry(QtCore.QRect(260, 70, 54, 16))
        self.cTimerValue.setObjectName("cTimerValue")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Timer.setText(_translate("MainWindow", "Timer"))
        self.Apply.setText(_translate("MainWindow", "Apply"))
        self.CurrentTimer.setText(_translate("MainWindow", "CurrentTimer"))
        self.cTimerValue.setText(_translate("MainWindow", "0"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
