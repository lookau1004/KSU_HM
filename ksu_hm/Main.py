import time
import cv2
from multiprocessing import Process,Value
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from UI import *

import numpy as np
import mediapipe as mp
import sys

global MainDict

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
        global MainDict
        self.configData.TimerNum = (int)(self.setTimer.toPlainText())
        self.configDict[('Config')] = self.configData
        MainDict = self.configDict        
        
        #print('ConfigData TimerNum = {}'.format(configDict[('Config')].TimerNum))
        
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
        
    def refresh_timer(self,second,sharedNum):
        sharedNum.value = second
        print("refrsh ",sharedNum.value)

class newCamara:


    def __init__(self,Second,sharedNum):
        self.CamaraOpen(Second,sharedNum)

    def CamaraOpen(self,Second,sharedNum):

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #BGR 이미지(opencv 기본)를 RGB 이미지로
                result = hands.process(frame)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                if result.multi_hand_landmarks is not None:
                    sharedNum.value = Second
                    for res in result.multi_hand_landmarks: # res 값 = landmark {x: y: z:}
                        joint = np.zeros((21,3))
                
                cv2.putText(frame, f'Timer: {int(sharedNum.value)}',(300,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    2,(255,0,0),3)
                cv2.imshow('Camera Window', frame)
            if cv2.waitKey(1) == 27: 
                break
            if (sharedNum.value == 0):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    MainDict = {}
    sharedNum = Value('i')

    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = ConfigWindow()
    ui.setupUi(mainWindow)
    ui.Apply.clicked.connect(ui.btnApply)
    mainWindow.show()
    app.exec_()

    setTime = MainDict[('Config')].TimerNum
    Second = int(setTime)      

    pTimer = Process(target = newTimer, name = "TimerProcess", args=(Second,sharedNum,))
    pCamera = Process(target = newCamara, name = "CameraProcess", args=(Second,sharedNum,))
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