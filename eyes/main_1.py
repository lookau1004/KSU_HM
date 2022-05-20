import cv2 as cv
import math
import numpy as np
import mediapipe as mp
import pyautogui

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]

LEFT_IRIS = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472] # 홍체 좌표
L_H_LEFT = [33]
L_H_RIGHT = [133] # 왼쪽 눈 좌우
R_H_LEFT = [362] #오른쪽 눈 좌우
R_H_RIGHT = [263]
mouse_current_position_x,mouse_current_position_y=pyautogui.position()
cap = cv.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces =1, 
    refine_landmarks = True, 
    min_detection_confidence =0.5,
    min_tracking_confidence=0.5
) as face_mesh:   # 얼굴 인식과 랜드마크 옵션 설정
   
    while True:
        ret,frame = cap.read()
        if not ret:
          break
        frame = cv.flip(frame,1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        img_h, img_w = frame.shape[:2]
        if results.multi_face_landmarks:
            # print(results.melti_face_landmarks[0].landmark)
            mesh_points=np.array([np.multiply([p.x,p.y],[img_w, img_h]).astype(int)for p in results.multi_face_landmarks[0].landmark])
            # print(mesh_points.shape)
            #cv.polylines(frame, [mesh_points[LEFT_IRIS]],True,(0,255,0),1,cv.LINE_AA) 
            #cv.polylines(frame, [mesh_points[RIGHT_IRIS]],True,(0,255,0),1,cv.LINE_AA) # 눈 인식 상태 사각형으로
            (l_cx, l_cy) , l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy) , l_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS]) # 홏체만의 좌표 출력 (소수점)
            center_left = np.array([l_cx,l_cy],dtype = np.int32) # 소수점을 정수화 홍체 중심
            center_right = np.array([r_cx,r_cy],dtype = np.int32)
            cv.circle(frame, center_left, int(l_radius),(255,0,255),1,cv.LINE_AA)
            cv.circle(frame, center_right, int(l_radius),(255,0,255),1,cv.LINE_AA)
            # print([center_left,center_right]) # 좌 우측 홍체 좌표

            x,y = center_left
            
            diff_x = x - mouse_current_position_x
            diff_y =y - mouse_current_position_y
            pyautogui.move((diff_x)*100, (diff_y)*100,_pause=False)      
            mouse_current_position_x,mouse_current_position_y= pyautogui.position()
                                                # _pause 옵션 끄면 렉 사라짐                                                                                            

        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
cap.release()
cv.destroyAllWindows
 
