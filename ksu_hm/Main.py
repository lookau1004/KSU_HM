import time
import cv2
from multiprocessing import Process,Value
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from UI import *

import numpy as np
import mediapipe as mp
import sys

global GlobalMainDict                           # 딕서녀리 전역 변수

class ConfigData():                             # 옵션 설정 데이터들을 클래스 형태로 정리
    def __init__(self):
        self.DefaultTimerNum = 10
    
    def Clear(self):
        self.DefaultTimerNum = 0

class ConfigWindow(Window.Ui_MainWindow):          # Window 클래스 PyQT5 상속 받아서 함수 추가 ( 수정 필요 )                                             
    def __init__(self):                            # Qt Designer로 디자인을 만든 후 ui 파일을  pyuic5 -x 이름.ui -o 이름.py 명령어 실행 후 py 파일로 바꿔줌
        super().__init__()                         # 부모 init() 실행
        self.configDict = {}                       # 딕셔너리 생성
        self.configDataClass = ConfigData()        # 데이터 클래스 생성

    def input_data(self):
        global GlobalMainDict                                                           # 전역 변수 사용
        self.configDataClass.DefaultTimerNum = (int)(self.WinTimerTxt.toPlainText())    # 데이터 클래스 안에 있는 기본 타이머 값에 윈도우 창에서 사용자가 입력한 값 대입
        self.configDict[('Config')] = self.configDataClass                              # 딕셔너리에 생성된 클래스를 저장
        GlobalMainDict = self.configDict                                                # 저장한 딕셔너리를 전역 딕셔너리에 대입
        
        #print('ConfigData TimerNum = {}'.format(configDict[('Config')].TimerNum))
        
    def btnApply(self):                                                                 # 확인 버튼을 눌렸을 때 실행 함수
        sharedNum.value = (int)(self.WinTimerTxt.toPlainText())                         # 공유 메모리 맵 value에 타이머 현재 값을 대입
        self.WinCurrentTimeLabel.setText((str)(sharedNum.value))                        # Win창에 있는 현재 타이머 표기 값 바꿈
        self.input_data()
        mainWindow.close()                                                              # 현재 윈폼 종료
        
        #print("윈도우에서 sharedNum 값 : " , sharedNum.value)

class newTimer:                                                                         # 타이머 클래스 ( 타이머에 관한 함수 포함 )
    def __init__(self,DefaultSecond,sharedNum):
        self.timer_run(DefaultSecond,sharedNum)
        
    def timer_run(self,DefaultSecond,sharedNum):                                        # 타이머 작동 함수 1초마다 값 줄어듬
        sharedNum.value = DefaultSecond
        while(sharedNum.value):
            sharedNum.value = sharedNum.value - 1
            time.sleep(1)
            print("Timer Running",sharedNum.value)
        
    def refresh_timer(self,DefaultSecond,sharedNum):                                    # 타이머 초기화 함수( 현재 사용 안함 )
        sharedNum.value = DefaultSecond
        print("refrsh ",sharedNum.value)

class newCamara:                                                                        # 카메라 클래스 ( 카메라 관련 함수 )

    def __init__(self,DefaultSecond,sharedNum):
        self.pTimer = Process(target = newTimer, name = "TimerProcess", args=(DefaultSecond,sharedNum,))        # 카메라 프로세스가 종료 했을때 타이머 프로세스도 종료 해야하므로 내부에서 선언
        self.pTimer.start()
        self.CamaraOpen(DefaultSecond,sharedNum)        

    def CamaraOpen(self,DefaultSecond,sharedNum):                                       # 카메라 메인 함수
        mp_hands = mp.solutions.hands                                                   # numpy hands
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)               # BGR 이미지(opencv 기본)를 RGB 이미지로
                result = hands.process(frame)                               # RGB값으로 바뀐 프레임에 손 모델 해석
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)               # 원 상태 복귀
                if result.multi_hand_landmarks is not None:                 # 결과값에 손이 있다면~
                    sharedNum.value = DefaultSecond                         # 타이머 초기화
                    for res in result.multi_hand_landmarks:                 # res 값 = landmark {x: y: z:}
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
        self.pTimer.terminate()                                             # 타이머 프로세스 강제종료

if __name__ == '__main__':
    
    GlobalMainDict = {}                                                     
    sharedNum = Value('i')                                                  # 프로세스간에 데이터 공유를 위해 Value를 이용하여 공유 메모리 맵 사용

    app = QtWidgets.QApplication(sys.argv)                                  # PyQT5 메인 윈도우 클래스 생성 부분
    mainWindow = QtWidgets.QMainWindow()
    ui = ConfigWindow()
    ui.setupUi(mainWindow)
    ui.WinApplyBtn.clicked.connect(ui.btnApply)
    mainWindow.show()
    app.exec_()

    DefaultSecond = int(GlobalMainDict[('Config')].DefaultTimerNum)

    pCamera = Process(target = newCamara, name = "CameraProcess", args=(DefaultSecond,sharedNum))

    pCamera.start()
    pCamera.join()

    print("현재 타이머 시간 : ",sharedNum.value)


# 1 윈도우 창안에서 값 교환 완료 , 데이터들을 클래스 형태로 정리한 후, 딕셔너리 형태로 변환 및 출력 완료
# 2. 윈도우 창 -> 다른 클래스 및 함수에 데이터 교환 해야함 ( 전역 변수로 함 )
# 3. 카메라 화면에 딜레이 숫자 표시
# 4. 손 잡혔을때 딜레이 초기화
# 5. 카메라 프로세스 종료 시 타이머 프로세스도 같이 종료
# ㅡㅡㅡㅡㅡㅡㅡㅡ 완료 ㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# 6. 윈도우 UI 작업을 클래스 내부 함수로 바꿀 것
# 7. 여러 해상도에서 UI가 크게 변하지 않을 것