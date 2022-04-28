import time
import cv2
from multiprocessing import Process,Value
from typing_extensions import Self
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from UI import *
import sys

class ConfigData():
    def __init__(self):
        self.TimerNum = 10
    
    def Clear(self):
        self.TimerNum = 0

class ConfigWindow(Window.Ui_MainWindow):     
    def __init__(self):
        super().__init__()
        self.configDict = {}
        self.configData = ConfigData()

    def input_data(self):
        self.configData.TimerNum = (int)(self.setTimer.toPlainText())
        self.configDict[('Config')] = self.configData
        print('ConfigData TimerNum = {}'.format(self.configDict[('Config')].TimerNum))
        return self.configDict

    def btnApply(self):
        sharedNum.value = (int)(self.setTimer.toPlainText())
        self.cTimerValue.setText((str)(sharedNum.value))
        print("윈도우에서 sharedNum 값 : " , sharedNum.value)
        self.input_data() 
        mainWindow.close()      

class newTimer:
    def __init__(self,second,sharedNum):
        self.second = second
        self.timer_run(self.second,sharedNum)
        
    def timer_run(self,Second,sharedNum):
        sharedNum.value = Second
        while(sharedNum.value):
            print('running...')
            sharedNum.value = sharedNum.value - 1
            time.sleep(1)
            print(sharedNum.value)
        
    def refresh_timer(self,sharedNum):
        sharedNum.value = 100
        print("refrsh ",sharedNum.value)

class newCamara:
    def __init__(self):
        self.CamaraOpen()

    def CamaraOpen(self):
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                cv2.imshow('Camera Window', frame)
            if cv2.waitKey(1) == 27: 
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    sharedNum = Value('i')

    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = ConfigWindow()
    ui.setupUi(mainWindow)
    ui.Apply.clicked.connect(ui.btnApply)
    mainWindow.show()
    app.exec_()

    setTime = input('시간을 입력하세요 .. (초 단위) :')
    Second = int(setTime)      

    pTimer = Process(target = newTimer, name = "TimerProcess", args=(Second,sharedNum,))
    pCamera = Process(target = newCamara, name = "CameraProcess" )
    #pc3 = Process(target = mainTimer.refresh_timer, args=[sharedNum])

    pTimer.start()
    pCamera.start()
    #pc3.start()

    pTimer.join()
    pCamera.join()
   # pc3.join()

    print(Second,sharedNum.value)


# 윈도우 창안에서 값 교환 완료 ,데이터들을 클래스 형태로 정리한 후, 딕셔너리 형태로 변환 및 출력 완료
# 1. 윈도우 창 -> 다른 클래스 및 함수에 데이터 교환 해야함
# 2. 카메라 화면에 딜레이 숫자 표시
# 3. 손 잡혔을때 딜레이 초기화