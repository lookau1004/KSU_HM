# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Window.ui'
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
        self.WinTimerLabel = QtWidgets.QLabel(self.centralwidget)
        self.WinTimerLabel.setGeometry(QtCore.QRect(20, 20, 41, 16))
        self.WinTimerLabel.setObjectName("WinTimerLabel")
        self.WinTimerTxt = QtWidgets.QTextEdit(self.centralwidget)
        self.WinTimerTxt.setGeometry(QtCore.QRect(70, 20, 111, 21))
        self.WinTimerTxt.setObjectName("WinTimerTxt")
        self.WinApplyBtn = QtWidgets.QPushButton(self.centralwidget)
        self.WinApplyBtn.setGeometry(QtCore.QRect(220, 20, 75, 23))
        self.WinApplyBtn.setObjectName("WinApplyBtn")
        self.WinCurrentTimeLabel = QtWidgets.QLabel(self.centralwidget)
        self.WinCurrentTimeLabel.setGeometry(QtCore.QRect(20, 70, 151, 31))
        self.WinCurrentTimeLabel.setObjectName("WinCurrentTimeLabel")
        self.WincTimerValue = QtWidgets.QLabel(self.centralwidget)
        self.WincTimerValue.setGeometry(QtCore.QRect(230, 80, 54, 16))
        self.WincTimerValue.setObjectName("WincTimerValue")
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
        self.WinTimerLabel.setText(_translate("MainWindow", "Timer"))
        self.WinApplyBtn.setText(_translate("MainWindow", "Apply"))
        self.WinCurrentTimeLabel.setText(_translate("MainWindow", "CurrentTimer"))
        self.WincTimerValue.setText(_translate("MainWindow", "0"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
