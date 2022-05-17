from UI import *

import time
import cv2
import pyautogui
import numpy as np
import mediapipe as mp
import sys
import os

from multiprocessing import Process,Value
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *

from tkinter import Frame
from cvzone.HandTrackingModule import HandDetector




global GlobalMainDict                           # 딕서녀리 전역 변수

gesture = {
    0:'start', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'seven', 8:'eight', 9:'nine', 10:'ten', 11: 'adios'}

gesture_1 = {1:'click', 3:'altright', 4:'altleft', 9:'spaceBar', 11: 'exit'}


class ConfigData():                             # 옵션 설정 데이터들을 클래스 형태로 정리
    def __init__(self):
        self.DefaultTimerNum = 10                                                                   # 기본 타이머 값
        self.DefaultPath = os.path.abspath(__file__)                                                # 현재 py 파일 경로
        self.CsvFilePath = self.DefaultPath.replace("Main.py","Data/gesture_train.csv")            # csv 파일 경로
    
    def Clear(self):
        self.DefaultTimerNum = 0
    
class ConfigWindow(Window.Ui_MainWindow):          # Window 클래스 PyQT5 상속 받아서 함수 추가 ( 수정 필요 )
    def __init__(self,mainWindow):                 # Qt Designer로 디자인을 만든 후 ui 파일을  pyuic5 -x 이름.ui -o 이름.py 명령어 실행 후 py 파일로 바꿔줌
        self.setup_UI(mainWindow)
        super().__init__()                         # 부모 init() 실행
        self.configDict = {}                       # 딕셔너리 생성
        self.configDataClass = ConfigData()        # 데이터 클래스 생성
        
    def setup_UI(self,mainWindow):                              # 윈도우 UI 생성 부분
        self.setupUi(mainWindow)                                # PyQT5(Window.py)의 setup Ui() 실행
        self.WinApplyBtn.clicked.connect(self.btnApply)         # 버튼에 함수 연결

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

class newTimer():                                                                         # 타이머 클래스 ( 타이머에 관한 함수 포함 )
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

class newCamara():                                                                        # 카메라 클래스 ( 카메라 관련 함수 )
    def __init__(self,DefaultSecond,sharedNum):
        self.configDataClass = ConfigData()                                                 # 데이터 클래스 생성
        self.pTimer = Process(target = newTimer, name = "TimerProcess", args=(DefaultSecond,sharedNum,))        # 카메라 프로세스가 종료 했을때 타이머 프로세스도 종료 해야하므로 내부에서 선언
        self.pTimer.start()
        self.CamaraOpen(DefaultSecond,sharedNum)
       
    def CamaraOpen(self,DefaultSecond,sharedNum):                                       # 카메라 메인 함수
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils                                              # numpy hands
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5, # 탐지 임계치
            min_tracking_confidence=0.5)  # 추적 임계치

        detector = HandDetector(maxHands=1)

        if sys.platform == "win32":
            cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            alt_command = 'alt'

        else:           
            cap = cv2.VideoCapture(0)  
            alt_command = 'command'
                        
        file = np.genfromtxt(self.configDataClass.CsvFilePath, delimiter=',') # 제스처 저장값 읽어오기
        angle = file[:,:-1].astype(np.float32) # 관절값만 추출 0 ~ 마지막 인덱스 전까지
        label = file[:,-1].astype(np.float32) # label 값만 추출, 마지막 인텍스만

        knn =cv2.ml.KNearest_create() #KNN 모델 초기화
        knn.train(angle,cv2.ml.ROW_SAMPLE,label) # KNN 학습
        
        is_Mode = False
        start_time_limit = time.time()
        gesture_n_times = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, }
        gesture_0_times = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, }
        mouse_current_position = {'x':0, 'y':0}

        while cap.isOpened():            
            success, frame = cap.read()

            if success:
                frame = cv2.flip(frame,1) # 좌우반전           
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)               # BGR 이미지(opencv 기본)를 RGB 이미지로
                result = hands.process(frame)                               # RGB값으로 바뀐 프레임에 손 모델 해석 ( 손의 위치와 관절 탐지 )
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)               # 원 상태 복귀

                if result.multi_hand_landmarks is not None:                 # 결과값에 손이 있다면~
                    sharedNum.value = DefaultSecond                         # 타이머 초기화
                    
                    for res in result.multi_hand_landmarks:                 # res 값 = landmark {x: y: z:}
                        joint = np.zeros((21, 3))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z]
                        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                        v = v2 - v1 # [20,3]
            
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                        angle = np.degrees(angle)

                        data = np.array([angle], dtype=np.float32)
                        ret, results, neighbours, dist = knn.findNearest(data, 3)
                        idx = int(results[0][0])
                        print(idx)

                        if(idx == 0) : # 시작 제스처일 경우
                            is_Mode = True
                            start_time_limit = time.time() + 0.5
                            print('start')
                            print('입력')
                        
                        if is_Mode and idx in gesture_1.keys(): # is_Mode = 시작 제스쳐 선입력 됐는지 확인
                            gesture_n_times[idx] += 1   # 제스쳐가 3번이상 인식 됐을때만, 아래 조건을 실행하게 합니다. 

                            if  (idx == 1) and gesture_n_times[idx] > 2:
                                pyautogui.click()
                                is_Mode = False
                                gestrue_n_times = gesture_0_times
                                break

                            elif (idx == 9) and gesture_n_times[idx] > 2:
                                pyautogui.press('space')
                                is_Mode = False
                                gestrue_n_times = gesture_0_times
                                break

                            elif (idx == 3) and gesture_n_times[idx] > 2:
                                pyautogui.hotkey(alt_command,'right')  # alt + 오른쪽 키 조합키 - 브라우저
                                pyautogui.press('right')         # 오른쪽 키 누르기 - 파워포인트
                                is_Mode = False
                                gestrue_n_times = gesture_0_times
                                break

                            elif (idx == 4) and gesture_n_times[idx] > 2:
                                pyautogui.hotkey(alt_command,'left')   # alt + 왼쪽 키 조합키 - 브라우저
                                pyautogui.press('left')           # 왼쪽 키 누르기 - 파워포인트
                                is_Mode = False
                                gestrue_n_times = gesture_0_times
                                break

                            elif (idx == 11) and gesture_n_times[idx] > 2:
                                sharedNum.value = 0
                                break

                        elif (idx == 1):    # 테스트기능) 시작제스쳐 없이, 1번 제스쳐의 검지 끝 좌표값으로 마우스 제어하기 
                            weight = 1 - abs(res.landmark[5].x - res.landmark[17].x)    # 화면과 손의 거리에 따라 가중치를 주기 위한 변수
                            diff_x = res.landmark[8].x - mouse_current_position['x']
                            diff_y = res.landmark[8].y - mouse_current_position['y']
                            mouse_current_position['x'] = res.landmark[8].x
                            mouse_current_position['y'] = res.landmark[8].y

                            if (abs(diff_x) + abs(diff_y)) > 0.25:  # 너무 많게는 포인터를 움직이지 않습니다.
                                pass

                            elif (abs(diff_x) + abs(diff_y)) > 0.005:   # 너무 적게는 포인터를 움직이지 않습니다.
                                pyautogui.move((diff_x)*2000**weight//1, (diff_y)*2000**weight//1,_pause=False)                          # _pause 옵션 끄면 렉 사라짐                                                                                            
                                gestrue_n_times = gesture_0_times                                                                        # (diff_x)*2000**weight//1 값 <= (diff_x)*2000//1 값
                        mp_drawing.draw_landmarks(frame,res,mp_hands.HAND_CONNECTIONS) # 관절을 프레임에 그린다.

                if start_time_limit < time.time():
                    is_Mode = False
                    gestrue_n_times = gesture_0_times
                    cv2.putText(frame, f'',(200,100),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,0,0),2)

                if(is_Mode):                                                                        # 입력 모드 체크
                    cv2.putText(frame, f'input mode',(200,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,\
                        1,(255,0,0),2)

                cv2.putText(frame, f'Timer: {int(sharedNum.value)}',(0,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,\
                    1,(255,0,0),2)
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
    ui = ConfigWindow(mainWindow)
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
# 6. 윈도우 UI 작업을 클래스 내부 함수로 바꿀 것
# ㅡㅡㅡㅡㅡㅡㅡㅡ 완료 ㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# 7. 여러 해상도에서 UI가 크게 변하지 않을 것