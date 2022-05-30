
from UI import *
import cv2
import mediapipe as mp
import numpy as np
import os, sys, subprocess
import os
import threading
import sys
import re                                               # 문자열 정규식

from pathlib import Path                                # 파일 찾는 라이브러리
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore

IsGetHand = False                                       # 카메라에 손이 보이는지 구별하는 Bool 변수
CamaraLoopOn = False                                    # 스레드 시작/종료 구별하는 Bool 변수  
IndexNumber = None                                      # 윈폼에서 가져오는 Index 넘버
 
class ConfigData():                                                                                        # 옵션 설정 데이터들을 클래스 형태로 정리
    def __init__(self):      
        self.DefaultPath = os.path.dirname(os.path.abspath(__file__))                                       # 현재 py 파일 경로        
        self.DataFolderPath = self.DefaultPath +"/Data/"                                                    # Data 폴더 경로
        self.CsvFilePath = self.DataFolderPath + "gesture_train.csv"                                        # csv 파일 경로
        self.TextFilePath = self.DataFolderPath + "labels.txt"                                              # labels 파일 경로
        self.NewCsvFileName = "new_gesture_train.csv"
        self.CamaraWidth = 640
        self.CamaraHeight = 480
        self.LabelNameDict = {}
    
    def Clear(self):
        self.CamaraWidth = 640
        self.CamaraHeight = 480

class TextFile():
    def __init__(self):
        self.configDataClass = ConfigData()      

    def SaveTextFile(self,label):                                                                               # labels.txt 파일 세이브
        global IndexNumber
        i = 0
        _str = ""
        self.LoadTextFile()
        while True:
            if i >= len(self.configDataClass.LabelNameDict):                                                    # 루프 i 값이 인덱스 키 값보다 많으면 탈출
                break;
            if i == int(IndexNumber):                                                                           # 윈폼에 적힌 인덱스에 해당하는 라벨은 윈폼에서 가져와서 입력
                _str += str(i) + " : " + label +"\n"                                                                                 
                i += 1
            else:
                _str += str(i) + " : " + self.configDataClass.LabelNameDict[i] +"\n"                            # 이전 테이터를 그대로 입력
                i += 1    
        file = open(self.configDataClass.TextFilePath,"w",encoding="utf-8")                                               
        file.write(str(_str))
        print("Label을 txt 파일에 저장했습니다")
        file.close()
        
    def LoadTextFile(self):                                                                                     # labels.txt 파일 로드
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
        try :
            if self.configDataClass.LabelNameDict.get(int(IndexNumber)):                                 # 저장한 딕셔너리에 IndexNumber 값이 있다면~
                return (self.configDataClass.LabelNameDict[int(IndexNumber)])                            # 해당하는 딕셔너리 값을 반환
        except:
            return None
        
class NewMainWindow(QtWidgets.QMainWindow):           # 기본 메인 윈도우 클래스의 closeEvent를 오버라이딩하기 위해서 클래스 생성
    def closeEvent(self,event):
        global CamaraLoopOn
        CamaraLoopOn = False

class ConfigWindow(wTraining.Ui_MainWindow):          # Window 클래스 PyQT5 상속 받아서 함수 추가 ( 수정 필요 )                                             
    def __init__(self,mainWindow):                    # Qt Designer로 디자인을 만든 후 ui 파일을  pyuic5 -x 이름.ui -o 이름.py 명령어 실행 후 py 파일로 바꿔줌
        self.configDataClass = ConfigData()           # 데이터 클래스 생성
        self.isTextFile()
        self.isCsvFile()
        self.TextFileClass = TextFile()               # TextFile 클래스 
        self.newMPClass = NewMediapipe()              # MP 클래스 생성             
        super().__init__()                            # 부모 init() 실행
        self.setup_UI(mainWindow)        
        self.configDict = {}                          # 딕셔너리 생성
        self.isThreadStarted = False                  # 스레스 중복 실행 방지용 변수
        self.isStop = False                           # exit했을때 저장하지 않기 위한 변수
        self.LabelName = ""
        
    def setup_UI(self,mainWindow):                                   # 윈도우 UI 생성 부분
        self.setupUi(mainWindow)                                     # PyQT5(Window.py)의 setup Ui() 실행
        self.WinStartBtn.clicked.connect(self.onStart)                 # 버튼에 start 함수 연결
        self.WinStopBtn.clicked.connect(self.onStop)                   # 버튼에 stop 함수 연결
        self.WinExitBtn.clicked.connect(self.onExit)                 # 버튼에 exit 함수 연결
        self.WinCaptureMotionBtn.clicked.connect(self.CaptureMotion)    # 버튼에 저장하는 함수 연결         
        self.WinOpenFolerBtn.clicked.connect(self.OpenFolder)        # 버튼에 폴더 여는 함수 연결
        self.WinLoadTextFileBtn.clicked.connect(self.LoadIndexWithDict)
        self.WinSaveTextFileBtn.clicked.connect(self.SaveIndewWithDict)

    def run(self):                                                   # 스레드로 돌릴 비디오 루프 함수 // 윈폼 라벨로 값을 넘겨 카메라를 보여줌
        global CamaraLoopOn 

        if sys.platform == "win32":
            cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.configDataClass.CamaraWidth)             # 카메라 해상도 조절
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.configDataClass.CamaraHeight)
        else:           
            cap = cv2.VideoCapture(0)      

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.WinCamaraLabel.resize(width, height)                        # 윈폼 라벨을 카메라 사이즈에 맞게 조정 width,height는 int형

        while CamaraLoopOn:
            success, frame = cap.read()
            if success:
                self.CheckCaptureMotion()
                frame = self.newMPClass.GraphicWithMp(frame)                  #MediaPipe 클래스 내에 있는 관절을 그려주는 함수로 값을 주고 받음
                h,w,c = frame.shape
                qImg = QtGui.QImage(frame.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                self.WinCamaraLabel.setPixmap(pixmap)
            else:
                print("cannot read frame.")
                break

        cap.release()
        if self.isStop:
            self.newMPClass.SaveFileToCsv()                                   # 루프에서 나오면 저장된 모션 값을 csv로 저장한다
        self.WinCamaraLabel.clear()                                      # 윈 라벨 초기화
        self.isStop = False

    def onStart(self):
        global CamaraLoopOn
        if not self.isThreadStarted:                                    # 작동중인 스레드가 없으면~
            CamaraLoopOn = True                                         # 카메라 루프를 돌려도 된다
            th = threading.Thread(target=self.run)                      # 윈폼 내 카메라 부분을 thread로 돌림
            th.start()
            self.isThreadStarted = True                                 # 스레드 작동중
            print("Thread started..")

    def onStop(self):
        global CamaraLoopOn
        global IsGetHand
        self.isStop = True                                              # stop을 눌러서 저장을 할것인가
        self.isThreadStarted = False                                    # stop 버튼을 누르면 작동중인 스레드가 없어질테니
        CamaraLoopOn = False                                            # 카메라 루프 탈출하고 (스레드가 무한루프에서 나옴)
        IsGetHand = False                                               # 혹시 손이 잡힌 상태로 stop을 눌렀다면 강제로 체크용 bool 값을 바꾸고
        self.CheckCaptureMotion()                                       # CaptionMotion 버튼의 활성화/비활성화를 isGetHand에 따라 바꾼다
        print("Thread stoped..")        

    def onExit(self):
        global CamaraLoopOn
        CamaraLoopOn = False                                            # 스레드를 무한루프에서 종료해야 프로그렘이 종료된다
        print("Program exit")
        sys.exit()

    def OpenFolder(self):                                               # CSV 폴더 여는 함수
        if sys.platform == "win32":
            os.startfile(self.configDataClass.DataFolderPath)
        else:                                               
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, self.configDataClass.DataFolderPath])

    def CheckCaptureMotion(self):                                          # 카메라 시작/중단 여부 체크 함수
        global IsGetHand
        if IsGetHand:
            self.WinCaptureMotionBtn.setEnabled(True)
        else :            
            self.WinCaptureMotionBtn.setEnabled(False)

    def LoadIndexWithDict(self):
        if self.input_index_data():
                self.LabelName = self.TextFileClass.LoadTextFile()
                self.WinTextLabelEdit.setText(self.LabelName)

    def SaveIndewWithDict(self):
        if self.input_index_data():
            try:
                if not int(IndexNumber) < 0:
                    self.TextFileClass.SaveTextFile(self.WinTextLabelEdit.text())
            except:
                print("label Save에서 에러")

    def CaptureMotion(self): 
        global IndexNumber                                                                     # 저장 버튼을 누르면 작동하는 함수 MediaPipe 클래스 안에 스택 함수를 사용
        if self.input_index_data():                                                            # 문자열에 값이 있다면~
            try:
                if  not int(IndexNumber) < 0 :                                                 # int형으로 변환 할 수 있는 문자열이면~   // 0을 넣으면 0<0 = False -> True // -1를 넣으면 True -> False
                    DataLinesInfo = self.newMPClass.StackToNp(IndexNumber)                     # NP 스택에 저장하고 File.shape 반환 (110,16)
                    StringLinesInfo = self.CvtDataToString(str(DataLinesInfo))                 # 반환된 값을 원하는 문자열 추가 후 String 형태로 변환
                    self.WinDataListWidget.insertItem(0,StringLinesInfo)                       # 윈폼 ListWidget에 아이템 추가
                    print("Motion 값이 저장되었습니다")
                else :
                    print("index란에 양수를 입력해주세요")
            except:
                print("index에서 에러가 발생했습니다")

    def input_index_data(self):    
        global IndexNumber                                                                      # 윈폼에 textline에 적힌 index 문자열 값을 가져오는 함수
        if self.WinIndexLineEdit.text() != '':                                                  # 문자열이 비어있지 않다면~
            IndexNumber = self.WinIndexLineEdit.text()
            return True
        else:
            print("index를 입력해주세요")
            return False

    def CvtDataToString(self,ConvertString):     
        global IndexNumber                                                                       # .shape 값을 String형으로 바꾸면서 필요한 문자열 추가
        ConvertString = ConvertString.replace(","," Total Lines")
        StringIndex = ConvertString.find(")")
        ConvertString = ConvertString[:StringIndex] + ' ea' + ConvertString[StringIndex:]
        ConvertString += " idx %s" %IndexNumber
        return ConvertString
        
    def isTextFile(self):                                                                       # labels.txt 파일이 없다면 기본값으로 생성
        isPath = Path(self.configDataClass.TextFilePath)
        if not isPath.exists():
            file = open(self.configDataClass.TextFilePath,"w",encoding="utf-8")
            file.write("1 : None")

    def isCsvFile(self):
        isPath = Path(self.configDataClass.CsvFilePath)
        if not isPath.exists():
            print("CSV 파일이 필요합니다")
            sys.exit()

class NewMediapipe(): 
    def __init__(self):    
        self.configDataClass = ConfigData()                                                      # 데이터 클래스 생성
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
        
        self.file = np.genfromtxt(self.configDataClass.CsvFilePath, delimiter=',')
    
    def StackToNp(self,IndexData):
        self.data = np.append(self.data, int(IndexData))
        self.file = np.vstack((self.file, self.data))
        return self.file.shape
    
    def SaveFileToCsv(self):      
        try:  
            np.savetxt(self.configDataClass.DataFolderPath + self.configDataClass.NewCsvFileName, self.file, fmt='%f',delimiter=',')
            print("CSV 파일로 저장되었습니다")
        except:
            print("저장에서 에러가 발생했습니다")
        
    def GraphicWithMp(self,img):        
        global IsGetHand
        img = cv2.flip(img, 1)    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        result = self.hands.process(img)

        if result.multi_hand_landmarks is not None:
            IsGetHand = True
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

        IsGetHand = False  
        return img

app = QtWidgets.QApplication(sys.argv)                                  # PyQT5 메인 윈도우 클래스 생성 부분
mainWindow = NewMainWindow()
ui = ConfigWindow(mainWindow)
mainWindow.show()
app.exec_()



# start을 눌러 카메라를 작동시키고 손이 있다면~ CaptureMotion를 누를 수 있다
# CaptureMotion를 누르면 현재 손 위치 값을 모아둔다
# stop을 누르면 모아둔 손 위치 값과 예전 csv파일의 값들을 합쳐 NewCsv 파일로 저장한다

# 라벨 로드는 파일에서 값을 불러와 인덱스에 해당하는 라벨명은 보여준다
# 라벨 저장은 파일에서 값을 불러와 윈폼에 적힌 라벨명을 해당하는 인덱스와 함께 이전 값과 함께 다시 저장
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ


# thread안에서 setEnabled/setEnabled 사용 시 timer 경고가 뜬다
# labels.txt에서 인덱스를 직접 적어서 늘여줘야 함