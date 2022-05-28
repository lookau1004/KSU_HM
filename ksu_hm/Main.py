from UI import *

import time
import cv2
import pyautogui
import numpy as np
import mediapipe as mp
import sys
import os
import re                                               # 문자열 정규식

from pathlib import Path                                # 파일 찾는 라이브러리
from multiprocessing import Process,Value
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

from ctypes import cast, POINTER                                # 볼륨 관련 모듈
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from tensorflow.keras.models import load_model

GlobalMainDict = {}                          # 딕서녀리 전역 변수 

gesture_1 = {1:'click', 3:'altright', 4:'altleft',6:'volumeUp',7:'volumeUp',9:'spaceBar',11:'exit', 12:'firstsave', 13:'secondsave', 14:'capture'}
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]

LEFT_IRIS = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472]

class ConfigData():                             # 옵션 설정 데이터들을 클래스 형태로 정리
    def __init__(self):
        self.DefaultTimerNum = 1                                                                   # 기본 타이머 값
        self.DefaultPath = os.path.dirname(os.path.abspath(__file__))                              # 현재 py 파일 경로
        self.DataFolderPath = self.DefaultPath + "/Data/"                                          # Data 폴더 경로
        self.CsvFilePath = self.DataFolderPath + "gesture_train.csv"                               # csv 파일 경로
        self.ImgFolderPath = self.DataFolderPath + "img/"
        self.TextFilePath = self.DataFolderPath + "labels.txt"
        self.TensorflowFilePath = self.DefaultPath + "/Tensorflow/KSU_model.h5"
        self.CamaraWidth = 640                                                                     # 640x480 | 480p | 4:3
        self.CamaraHeight = 480
        self.LabelNameDict = {}

        devices = AudioUtilities.GetSpeakers()                                                     # 볼륨 관련 모듈 초기화
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.volumeRange = self.volume.GetVolumeRange()

    def Clear(self):
        self.DefaultTimerNum = 0

class NewMainWindow(QtWidgets.QMainWindow):           # 기본 메인 윈도우 클래스의 오버라이딩하기 위해서 클래스 생성 
    def closeEvent(self,event):                       # 종료 시 이벤트 ( 현재 기능 없음 )
        pass            
    
class ConfigWindow(Window.Ui_MainWindow):          # Window 클래스 PyQT5 상속 받아서 함수 추가 ( 수정 필요 )
    def __init__(self,mainWindow):                 # Qt Designer로 디자인을 만든 후 ui 파일을  pyuic5 -x 이름.ui -o 이름.py 명령어 실행 후 py 파일로 바꿔줌
        super().__init__()                         # 부모 init() 실행
        self.setup_UI(mainWindow)
        self.configDict = {}                       # 딕셔너리 생성
        self.configDataClass = ConfigData()        # 데이터 클래스 생성
        self.isCsvFile()
        self.isTextFile()

    def setup_UI(self,mainWindow):                              # 윈도우 UI 생성 부분
        self.setupUi(mainWindow)                                # PyQT5(Window.py)의 setup Ui() 실행
        self.WinApplyBtn.clicked.connect(self.btnApply)         # 버튼에 함수 연결
        self.WinTimerTxt.returnPressed.connect(self.btnApply)

    def input_data(self):
        global GlobalMainDict                                                           # 전역 변수 사용
        self.configDataClass.DefaultTimerNum = int(self.WinTimerTxt.text())           # 데이터 클래스 안에 있는 기본 타이머 값에 윈도우 창에서 사용자가 입력한 값 대입
        self.configDict['Config'] = self.configDataClass                              # 딕셔너리에 생성된 클래스를 저장
        GlobalMainDict = self.configDict                                                # 저장한 딕셔너리를 전역 딕셔너리에 대입
        
        #print('ConfigData TimerNum = {}'.format(configDict[('Config')].TimerNum))
        
    def btnApply(self):                                                                 # 확인 버튼을 눌렸을 때 실행 함수
        try:
            if not int(self.WinTimerTxt.text()) < 0:                                            # 문자열 및 음수 체크
                sharedNum.value = int(self.WinTimerTxt.text())                                # 공유 메모리 맵 value에 타이머 현재 값을 대입
                self.WinCurrentTimeLabel.setText(str(sharedNum.value))                        # Win창에 있는 현재 타이머 표기 값 바꿈
                self.input_data()
        except:
            print("타이머 입력 에러")
        mainWindow.close()                                                              # 현재 윈폼 종료
        
        #print("윈도우에서 sharedNum 값 : " , sharedNum.value)

    def isTextFile(self):                                                                       # labels.txt 파일이 없다면 기본값으로 생성
        isPath = Path(self.configDataClass.TextFilePath)
        if not isPath.exists():
            file = open(self.configDataClass.TextFilePath,"w",encoding="utf-8")
            file.write("1 : None")
            sys.exit()

    def isCsvFile(self):
        isPath = Path(self.configDataClass.CsvFilePath)
        if not isPath.exists():
            print("CSV 파일이 필요합니다")
            sys.exit()

class newTimer():                                                                         # 타이머 클래스 ( 타이머에 관한 함수 포함 )
    def __init__(self,DefaultSecond,sharedNum):
        self.timer_run(DefaultSecond,sharedNum)
        
    def timer_run(self,DefaultSecond,sharedNum):                                        # 타이머 작동 함수 1초마다 값 줄어듬
        sharedNum.value = DefaultSecond
        while(sharedNum.value):
            time.sleep(1)
            sharedNum.value = sharedNum.value - 1
            print("Timer Running",sharedNum.value)
        
    def refresh_timer(self,DefaultSecond,sharedNum):                                    # 타이머 초기화 함수( 현재 사용 안함 )
        sharedNum.value = DefaultSecond
        print("refrsh ",sharedNum.value)

class newCamara():                                                                        # 카메라 클래스 ( 카메라 관련 함수 )
    def __init__(self,DefaultSecond,sharedNum):
        self.configDataClass = ConfigData()                                                 # 데이터 클래스 생성
        self.LoadLabelFile()
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

        if sys.platform == "win32":
            cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.configDataClass.CamaraWidth)             # 카메라 해상도 조절
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.configDataClass.CamaraHeight)
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
        is_TimeCopy = False
        TensorGestureMode = False

        start_time_limit = time.time()
        gesture_n_times = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, }
        gesture_0_times = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, }

        volumeValue = self.configDataClass.volume.GetMasterVolumeLevel()

        mouse_current_position = {'x':0, 'y':0}
        mouse_down = False 
        mouse_drag = {'x1':0, 'y1':0, 'x2':0, 'y2':0, }
        mouse_capture = {'x1':0, 'y1':0, 'x2':0, 'y2':0, }

        seq = []
        action_seq = []
        actions = ['left', 'right', 'zoomin']
        self.this_action = ""
        seq_length = 30
        model = load_model(self.configDataClass.TensorflowFilePath)

        while cap.isOpened():            
            success, frame = cap.read()
            idx = None

            if success:
                frame = cv2.flip(frame,1) # 좌우반전           
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)               # BGR 이미지(opencv 기본)를 RGB 이미지로
                result = hands.process(frame)                               # RGB값으로 바뀐 프레임에 손 모델 해석 ( 손의 위치와 관절 탐지 )
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)               # 원 상태 복귀
                
                if result.multi_hand_landmarks is not None:                 # 결과값에 손이 있다면~
                    sharedNum.value = DefaultSecond                         # 타이머 초기화           
                    start_time_limit = time.time() + 10                     # 10초 후에 저장값 리셋         

                    for res in result.multi_hand_landmarks:                 # res 값 = landmark {x: y: z:}
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                        v = v2 - v1 # [20,4]
            
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                        angle = np.degrees(angle)    
              
                        data = np.array([angle], dtype=np.float32)
                        ret, results, neighbours, dist = knn.findNearest(data, 3)
                        idx = int(results[0][0])

                        if(idx == 0) : # 0번 인덱스일때 텐서 플로우 모드 켜짐
                            is_TimeCopy = True                                  # 지금 타임을 저장하였고
 
                            if TensorGestureMode == False:                      # 텐서플로우 모드가 종료되어있으면
                                gesture_n_times[0] += 1                         # 저장값을 증가하고
                                if(gesture_n_times[0] > 30):                                
                                    gesture_n_times[0] = 0                      # 초기화
                                    TensorGestureMode = True                    # 제스처 모드 온

                        if(idx == 3):                                           # 3번 인덱스면 텐서 플로우 모드 종료
                            gesture_n_times[3] += 1
                            if gesture_n_times[3] > 30:                         # 3번 인덱스가 계속 잡히면
                                gesture_n_times[3] = 0
                                TensorGestureMode = False                       # 텐서플로우 모드 종료

                        if TensorGestureMode == True:
                            if not self.this_action == "":
                                cv2.putText(frame, self.this_action,(400,200),cv2.FONT_HERSHEY_COMPLEX_SMALL,                                
                                1,(0,0,255),2)  

                            d = np.concatenate([joint.flatten(), angle]) # flatten() = 다차원 배열 1차원 배열 변환 // concatenate() numpy의 배열 연결 함수
                            seq.append(d)

                            if len(seq) >= seq_length:
                                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0) # expand_dims() 배열에 차원 추가
                                y_pred = model.predict(input_data).squeeze()                                       # squeeze() 차원을 압축한다 + 1의 중복 제거// predict() 모델의 신뢰도가 배열로 보여짐
                                i_pred = int(np.argmax(y_pred))                                                    # argmax() 배열에서 최대값 찾는 함수
                                conf = y_pred[i_pred]

                                if conf >= 0.9:                                                                     # 신뢰도가 90% 이상
                                    action = actions[i_pred]
                                    action_seq.append(action)

                                    if len(action_seq) >= 3:
                                        self.this_action = '?'
                                        if action_seq[-1] == action_seq[-2] == action_seq[-3]:                     # 손 모양 값이 연속되면
                                            self.this_action = action
                                            print("들어온 제스처 값 : " + self.this_action)
                                            action_seq.clear()  
            
                        if is_Mode and idx in gesture_1.keys(): # is_Mode = 시작 제스쳐 선입력 됐는지 확인
                            gesture_n_times[idx] += 1   # 제스쳐가 3번이상 인식 됐을때만, 아래 조건을 실행하게 합니다. 

                            if (idx == 1) and gesture_n_times[idx] > 2:
                                pyautogui.hotkey(alt_command,'right')  # alt + 오른쪽 키 조합키 - 브라우저
                                pyautogui.press('right')         # 오른쪽 키 누르기 - 파워포인트
                                is_Mode = False
                                gesture_n_times = gesture_0_times
                                break

                            elif (idx == 2) and gesture_n_times[idx] > 2:
                                pyautogui.hotkey(alt_command,'left')   # alt + 왼쪽 키 조합키 - 브라우저
                                pyautogui.press('left')           # 왼쪽 키 누르기 - 파워포인트
                                is_Mode = False
                                gesture_n_times = gesture_0_times
                                break

                            elif (idx == 11) and gesture_n_times[idx] > 2:
                                sharedNum.value = 0
                                break                          

                        if TensorGestureMode == False:
                            if (idx == 5):
                                face_mesh = mp_face_mesh.FaceMesh(max_num_faces =1, refine_landmarks = True, min_detection_confidence =0.5,min_tracking_confidence=0.5 )  # 얼굴 인식과 랜드마크 옵션 설정
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = face_mesh.process(rgb_frame)
                                img_h, img_w = frame.shape[:2]
                                if results.multi_face_landmarks:
                                    mesh_points=np.array([np.multiply([p.x,p.y],[img_w, img_h]).astype(int)for p in results.multi_face_landmarks[0].landmark])  
                                    (l_cx, l_cy) , l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                                    (r_cx, r_cy) , l_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                                    center_left = np.array([l_cx,l_cy],dtype = np.int32) # 소수점을 정수화 홍체 중심
                                    center_right = np.array([r_cx,r_cy],dtype = np.int32)
                                    cv2.circle(frame, center_left, int(l_radius),(255,0,255),1,cv2.LINE_AA)
                                    cv2.circle(frame, center_right, int(l_radius),(255,0,255),1,cv2.LINE_AA)
                                    x,y = center_left
                
                                    diff_x = x - mouse_current_position['x']
                                    diff_y =y - mouse_current_position['y']
                                    mouse_current_position['x'] = x 
                                    mouse_current_position['y'] = y
                                    pyautogui.move((diff_x), (diff_y),_pause=False)   # _pause 옵션 끄면 렉 사라짐                                                                                            
                                    gesture_n_times[5] = 0       
                            
                            elif (idx == 6):                                                                                                                 # 윈도우 -37 = 0    // 0 == 100    
                                gesture_n_times[6] += 1                                                                                                             # 무한 루프 돌때마다 +1
                                if gesture_n_times[6] > 7:                                                                                                          # 7장 돌면
                                    volumeValue += 1                                                                                                         # 현재 볼륨값에 +1
                                    if volumeValue <= self.configDataClass.volumeRange[2] - 1 and volumeValue >= self.configDataClass.volumeRange[0] + 1:    # 컴퓨터 볼륨 범위에 에러 안전범위까지 더해서~
                                        self.configDataClass.volume.SetMasterVolumeLevel(volumeValue, None)                                                  # 볼륨 값을 대입한다
                                        print("volume UP")
                                    gesture_n_times[6] = 0                                                                                                          # 루프 개수 초기화                                            
                                break

                            elif (idx == 7):
                                gesture_n_times[7] -= 1                                                                                           
                                if gesture_n_times[7] < -7:
                                    volumeValue -= 1
                                    gesture_n_times[7] = 0
                                    if volumeValue >= self.configDataClass.volumeRange[0] + 1 and volumeValue <= self.configDataClass.volumeRange[2] - 1:
                                        self.configDataClass.volume.SetMasterVolumeLevel(volumeValue, None) 
                                        print("volume Down")
                                    gesture_n_times[7] = 0  
                                break

                            elif (idx == 3):                                                                     # 테스트기능) 시작제스쳐 없이, 1번 제스쳐의 검지 끝 좌표값으로 마우스 제어하기                                                     
                                #weight = 1 - abs(res.landmark[5].x - res.landmark[17].x)                                                    # 화면과 손의 거리에 따라 가중치를 주기 위한 변수
                                diff_x = res.landmark[9].x - mouse_current_position['x']
                                diff_y = res.landmark[9].y - mouse_current_position['y']
                                mouse_current_position['x'] = res.landmark[9].x
                                mouse_current_position['y'] = res.landmark[9].y

                                if (abs(diff_x) + abs(diff_y)) > 0.25:                                                                       # 너무 많게는 포인터를 움직이지 않습니다.
                                    pass

                                elif (abs(diff_x) + abs(diff_y)) > 0.003:                                                                    # 너무 적게는 포인터를 움직이지 않습니다.
                                    pyautogui.move((diff_x)*2000//1, (diff_y)*2000//1,_pause=False)                                          # _pause 옵션 끄면 렉 사라짐                                                                                            
                                    #gesture_n_times = gesture_0_times                                                                       # (diff_x)*2000**weight//1 값 <= (diff_x)*2000//1 값
                                    
                                # 아래가 마우스 기능 
                                if abs(res.landmark[10].y - res.landmark[12].y)  < 0.055 :  # 우클릭 
                                    pyautogui.click(button = 'right')
                                
                                if res.landmark[4].x > res.landmark[3].x:   # 스페이스바 
                                    pyautogui.press('space')    
                                
                                if abs(res.landmark[8].y - res.landmark[6].y) < 0.06: # 좌클릭 (마우스 다운)
                                    if not mouse_down:  
                                        pyautogui.mouseDown() 
                                        mouse_down = True 
                                        mouse_drag['x1'], mouse_drag['y1'] = pyautogui.position()
                                    else:
                                        pass
                                
                                elif mouse_down:    #   좌클릭 (마우스 업: 위의 조건이 만족하지 않을때 실행 = 손 가락 펼침)
                                    pyautogui.mouseUp()
                                    mouse_drag['x2'], mouse_drag['y2'] = pyautogui.position()
                                    print("x diff>", mouse_drag['x2'] - mouse_drag['x1'])
                                    print("y diff>", mouse_drag['y2'] - mouse_drag['y1'])
                                    mouse_down = False 
                                break

                            elif (idx == 12):
                                #마우스 위치1 x,y 좌표 저장
                                mouse_capture['x1'] = mouse_current_position['x']
                                mouse_capture['y1'] = mouse_current_position['y']
                                break

                            elif (idx == 13):
                                #마우스 위치2 x,y 좌표 저장
                                mouse_capture['x2'] = mouse_current_position['x']
                                mouse_capture['y2'] = mouse_current_position['y']
                                break

                            elif (idx == 14):
                                #마우스 위치 1, 2 사이의 사각형 캡쳐
                                pyautogui.screenshot('my_region.png', region=(mouse_capture['x1'], mouse_capture['y1'], mouse_capture['x2'], mouse_capture['y2']))

###################################  for res in result.multi_hand_landmarks: 끝  ###################################################################

                    mp_drawing.draw_landmarks(frame,res,mp_hands.HAND_CONNECTIONS)                                                   # 관절을 프레임에 그린다.        

###################################  if result.multi_hand_landmarks is not None: 끝  ###############################################################

                if start_time_limit < time.time() and is_TimeCopy == True:
                    is_TimeCopy = False
                    gesture_n_times = gesture_0_times.copy()
                    print("gesture_n_times reset")

                if(TensorGestureMode):                                                                                                            # 입력 모드 체크
                    cv2.putText(frame, f'Mode ON',(200,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,                                           # 화면에 input mode 표시
                        1,(0,0,255),2) # 빨강                                                                                            # (0,0,255) Blue,Green,Red 순서

                cv2.putText(frame, f'Timer: {int(sharedNum.value)}',(0,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,                              # 화면에 타이머 표시
                    1,(192,192,192),2) # 은색
                
                if not idx == None:                                                                                                     # 관절을 입힌 프레임을 숫자 이미지를 추가하는 함수에 전달
                    frame = self.AddIdxToFrame(idx,frame)
                    cv2.putText(frame,self.configDataClass.LabelNameDict[idx],(400,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,                  # 화면에 라벨명 표시함, 유니코드 지원 안함, 한글 라벨은 표시 불가
                    1,(0,255,0),2) # 초록색                                                                                                      

                cv2.imshow('Camera Window', frame)                

###################################  if success: 끝  ##############################################################################################

            if cv2.waitKey(1) == 27:
                break
           
            if (sharedNum.value == 0):
                break

###################################  while cap.isOpened(): 끝  #####################################################################################   
  
        cap.release()
        cv2.destroyAllWindows()
        self.pTimer.terminate()                                                                                                       # 타이머 프로세스 강제종료

    def AddIdxToFrame(self,_idx,_frame):                                                                        # Index 이미지 파일을 frame에 합쳐주는 함수
        if _idx >= 0 and _idx <= 9:
            NumImg = cv2.imread(self.configDataClass.ImgFolderPath + str(_idx)+".png")                                                            
            h_NumImg, w_NumImg, _ = NumImg.shape
            h_frame, w_frame, _ = _frame.shape

            center_y = int(h_frame / 6)
            center_x = int(w_frame / 6)

            top_y = center_y - int(h_NumImg / 2)
            left_x = center_x - int(w_NumImg / 2)

            bottom_y = top_y + h_NumImg
            right_x = left_x + w_NumImg

            _frame[top_y:bottom_y,left_x:right_x] = NumImg

        return _frame

    def LoadLabelFile(self):                                                                                     # labels.txt 파일 로드
        i = 0   
        file = open(self.configDataClass.TextFilePath,"r",encoding="utf-8")

        while True:
            line = file.readline()
            if not line:
                break            
            _str = line.replace(":","")                                                                  # : 삭제
            _str = re.sub(r"[0-9]","",_str)                                                              # 정규식으로 숫자 삭제            
            _str = _str.strip()                                                                          # 개행 삭제
            self.configDataClass.LabelNameDict[i] = _str
            i += 1

        file.close()    

if __name__ == '__main__':

    sharedNum = Value('i')                                                  # 프로세스간에 데이터 공유를 위해 Value를 이용하여 공유 메모리 맵 사용

    app = QtWidgets.QApplication(sys.argv)                                  # PyQT5 메인 윈도우 클래스 생성 부분
    mainWindow = NewMainWindow()
    ui = ConfigWindow(mainWindow)
    mainWindow.show()
    app.exec_()

    if 'Config' in GlobalMainDict:
        DefaultSecond = int(GlobalMainDict['Config'].DefaultTimerNum)
        pCamera = newCamara(DefaultSecond,sharedNum)

# 1 윈도우 창안에서 값 교환 완료 , 데이터들을 클래스 형태로 정리한 후, 딕셔너리 형태로 변환 및 출력 완료
# 2. 윈도우 창 -> 다른 클래스 및 함수에 데이터 교환 해야함 ( 전역 변수로 함 )
# 3. 카메라 화면에 딜레이 숫자 표시
# 4. 손 잡혔을때 딜레이 초기화
# 5. 카메라 프로세스 종료 시 타이머 프로세스도 같이 종료
# 6. 윈도우 UI 작업을 클래스 내부 함수로 바꿀 것
# ㅡㅡㅡㅡㅡㅡㅡㅡ 완료 ㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# 7. 여러 해상도에서 UI가 크게 변하지 않을 것