import cv2
from cv2 import THRESH_BINARY
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
detector = dlib.get_frontal_face_detector()
predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint (p1, p2): # 눈 위아래 길이 
    return int((p1.x+p2.x)/2),int((p1.y + p2.y)/2) # 소수점 방지를 위한 int


def get_blinking_ratio(eye_points, landmarks):
        left_point = (landmarks.part(eye_points[0]).x,landmarks.part(eye_points[0]).y )  # 왼쪽눈 왼쪽
        right_point = (landmarks.part(eye_points[3]).x,landmarks.part(eye_points[3]).y ) # 왼쪽눈 오른쪽

        center_top = midpoint(landmarks.part(eye_points[1]),landmarks.part(eye_points[2]))
        center_bottom = midpoint(landmarks.part(eye_points[5]),landmarks.part(eye_points[4]))
        # hor_line =cv2.line(frame, left_point, right_point, (0,255,0),2)
        # hor_line =cv2.line(frame, center_top, center_bottom, (0,255,0),2)
        hor_line_lenght =hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_lenght = hypot((center_top[0]-center_bottom[0]), (center_top[1]-center_bottom[1]))
        ratio= hor_line_lenght/ver_line_lenght   
        return ratio
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces =detector(gray)
    for face in faces:
        #x, y =face.left() , face.top()
        #x1, y1 =face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1) , (0,255,0),2) 얼굴 사각형으로 인식
        # 감빡임 인식 
        landmarks =predictor(gray, face)
        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47], landmarks)

        # Geze사용
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        # cv2.polylines(mask, [left_eye_region, True, (0,0,255, 2])
        
        
       
        hight, width, _ = frame.shape
        mask = np.zeros((hight, width), np.uint8)

        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray,gray,mask=mask)

        min_x = np.min(left_eye_region[ :, 0])
        max_x = np.max(left_eye_region[ :, 0])
        min_y = np.min(left_eye_region[ :, 1])
        max_y = np.max(left_eye_region[ :, 1])


        #eye = frame [min_y: max_y, min_x: max_x]
        gray_eye = left_eye [min_y: max_y, min_x: max_x]
        #gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        # a, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        a, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        threshold_eye = cv2.resize(threshold_eye,None, fx =10, fy =10 )
        #eye = cv2.resize(eye,None, fx =10, fy =10 )
        eye = cv2.resize(gray_eye,None, fx =10, fy =10 )
        # cv2.imshow("eye",eye)
        # cv2.imshow("threshold",threshold_eye)
        cv2.imshow("left_eye",left_eye)
    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)
    if key ==27:
        break
cap.release()
cv2.destroyAllWindows()
