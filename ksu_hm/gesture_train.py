import cv2
import mediapipe as mp
import numpy as np
from UI import *

import threading
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore

IsCamara = False     

class ConfigData():                             # 옵션 설정 데이터들을 클래스 형태로 정리
    def __init__(self):
        self.DefaultTimerNum = 10        
        self.IndexData = None
    
    def Clear(self):
        self.DefaultTimerNum = 0
        self.IndexData = None

class ConfigWindow(wTraining.Ui_MainWindow):          # Window 클래스 PyQT5 상속 받아서 함수 추가 ( 수정 필요 )                                             
    def __init__(self,mainWindow):                 # Qt Designer로 디자인을 만든 후 ui 파일을  pyuic5 -x 이름.ui -o 이름.py 명령어 실행 후 py 파일로 바꿔줌
        self.setup_UI(mainWindow)
        super().__init__()                         # 부모 init() 실행
        self.configDict = {}                       # 딕셔너리 생성
        self.configDataClass = ConfigData()        # 데이터 클래스 생성
        self.newMP = NewMediapipe()
        self.running = False   
        
    def setup_UI(self,mainWindow):                              # 윈도우 UI 생성 부분
        self.setupUi(mainWindow)                                # PyQT5(Window.py)의 setup Ui() 실행
        self.WinStartBtn.clicked.connect(self.start)            # 버튼에 함수 연결
        self.WinStopBtn.clicked.connect(self.stop)              # 버튼에 함수 연결
        self.WinExitBtn.clicked.connect(self.onExit)            # 버튼에 함수 연결
        self.WinCaptureMotionBtn.clicked.connect(self.SaveMotion)
        self.WinCaptureMotionBtn.setDisabled(True)        

    def run(self):
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.WinCamaraLabel.resize(width, height)

        while self.running:
            ret, img = cap.read()
            if ret:
                self.CheckCamara()
                img = self.newMP.GraphicWithMp(img)
                h,w,c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                self.WinCamaraLabel.setPixmap(pixmap)
            else:
                print("cannot read frame.")
                break
        cap.release()
        self.newMP.SaveFileToCsv()
        print("Thread end.")
        self.WinCamaraLabel.clear()

    def CheckCamara(self):
        global IsCamara
        if IsCamara:
            self.WinCaptureMotionBtn.setEnabled(True)
        else :            
            self.WinCaptureMotionBtn.setDisabled(True)

    def start(self):
        self.running = True
        th = threading.Thread(target=self.run)
        th.start()
        print("started..")

    def stop(self):
        self.running = False
        print("stoped..")        

    def onExit(self):
        print("exit")
        self.stop()
        sys.exit()

    def SaveMotion(self):
        if self.input_index_data():
            try:
                if int(self.configDataClass.IndexData):
                    self.newMP.StackToNp(self.configDataClass.IndexData)
            except:
                print("index란에 숫자를 입력해주세요")

    def input_index_data(self):
        if self.WinIndexLineEdit.text() != '':
            self.configDataClass.IndexData = self.WinIndexLineEdit.text()
            return True
        else:
            print("index를 입력해주세요")
            return False

class NewMediapipe(): 
    def __init__(self):    
        self.max_num_hands = 1        
        self.gesture = {
            0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
            6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'fy'
        }        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.file = np.genfromtxt('ksu_hm/Data/gesture_train.csv', delimiter=',')
    
    def StackToNp(self,IndexData):
        self.data = np.append(self.data, int(IndexData))
        self.file = np.vstack((self.file, self.data))
        print(self.file.shape)
    
    def SaveFileToCsv(self):        
        np.savetxt('ksu_hm/Data/new_gesture_train.csv', self.file, fmt='%f',delimiter=',')

    def GraphicWithMp(self,img):        
        global IsCamara
        img = cv2.flip(img, 1)    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        result = self.hands.process(img)

        if result.multi_hand_landmarks is not None:
            IsCamara = True
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                self.angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                self.angle = np.degrees(self.angle) # Convert radian to degree
                
                self.data = np.array([self.angle], dtype=np.float32)
                self.mp_drawing.draw_landmarks(img, res, self.mp_hands.HAND_CONNECTIONS)   
                return img   
        IsCamara = False  
        return img

app = QtWidgets.QApplication(sys.argv)                                  # PyQT5 메인 윈도우 클래스 생성 부분
mainWindow = QtWidgets.QMainWindow()
ui = ConfigWindow(mainWindow)
mainWindow.show()
app.exec_()


# thread안에서 setEnabled/setDisabled 사용 시 timer 경고가 뜬다