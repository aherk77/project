## -*- coding: utf-8 -*-
import numpy as np
import cv2
import mediapipe as mp
import pyautogui
import math

# 손가락 움직임 및 손목 움직임을 판단하기 위한 임계값 (조정 가능)
fingertip_movement_threshold = 8 # 임계값 이상 손가락 움직임 판단을 위한 값 (픽셀 단위)
wrist_movement_threshold = 3  # 임계값 이상 손목 움직임 판단을 위한 값 (픽셀 단위)
radius = 15 

key_coordinates = {
    '`': (576, 415),
    '1': (540, 415),
    '2': (505, 415),
    '3': (470, 415),
    '4': (436, 415),
    '5': (402, 415),
    '6': (368, 415),
    '7': (334, 416),
    '8': (300, 415),
    '9': (266, 415),
    '0': (232, 416),
    '-': (199, 416),
    '=': (166, 416),
    'backspace': (130, 416),
    'tab': (575, 378),
    'q': (539, 379),
    'w': (504, 378),
    'e': (468, 378),
    'r': (435, 379),
    't': (401, 378),
    'y': (367, 379),
    'u': (333, 379),
    'i': (300, 379),
    'o': (266, 380),
    'p': (231, 380),
    '[': (198, 380),
    ']': (164, 380),
    '\\': (130, 381),
    'capslock': (574, 341),
    'a': (539, 341),
    's': (504, 341),
    'd': (468, 342),
    'f': (434, 342),
    'g': (400, 343),
    'h': (367, 343),
    'j': (332, 344),
    'k': (299, 344),
    'l': (265, 344),
    ';': (230, 345),
    "'": (198, 345),
    'z': (501, 306),
    'x': (468, 307),
    'c': (434, 307),
    'v': (399, 308),
    'b': (365, 308),
    'n': (331, 309),
    'm': (298, 309),
    ',': (263, 309),
    '.': (230, 310),
    '/': (196, 309),
    'ctrlleft': (573, 268),
    'winleft': (537, 269),
    'alt': (501, 270),
    'hangul': (229, 273),
    'winright': (196, 275),
    'fn': (160, 275),
    'ctrlright': (126, 275)
}

def detect_hands():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    # 손가락 및 손목 이전 프레임 좌표 초기화
    R_fingertips = None 
    L_fingertips = None
    Right_wrist = None
    Left_wrist = None
    prev_R_fingertip = None
    prev_L_fingertip = None
    prev_R_wrist = None
    prev_L_wrist = None 

    # 손 감지 모델 초기화
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 비디오 캡처 시작
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        # 이미지를 RGB로 변환하여 손 감지 수행
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        cv2.rectangle(image, (258, 258), (469, 282), (255, 255, 255), 1) #space
        cv2.putText(image, 'space', (363, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)

        cv2.rectangle(image, (120, 298), (171, 322), (255, 255, 255), 1) #shift right
        cv2.putText(image, 'shift', (145, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)

        cv2.rectangle(image, (530, 293), (578, 317), (255, 255, 255), 1) #shift left
        cv2.putText(image, 'shift', (554, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)

        cv2.rectangle(image, (121, 334), (168, 357), (255, 255, 255), 1) #enter
        cv2.putText(image, 'enter', (145, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)

        for key, coord in key_coordinates.items():
            cv2.circle(image, coord, radius, (255, 255, 255), 1)
            cv2.putText(image, key, (coord[0], coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)

        # 손 감지 결과를 받아와서 처리
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y))

                if landmarks[20][0] < landmarks[4][0]:
                     R_fingertips = [(int(landmarks[4][0] * image.shape[1]), int(landmarks[4][1] * image.shape[0])),  # 엄지 끝점
                                        (int(landmarks[8][0] * image.shape[1]), int(landmarks[8][1] * image.shape[0])),  # 검지 끝점
                                        (int(landmarks[12][0] * image.shape[1]), int(landmarks[12][1] * image.shape[0])),  # 중지 끝점
                                        (int(landmarks[16][0] * image.shape[1]), int(landmarks[16][1] * image.shape[0])),  # 약지 끝점
                                        (int(landmarks[20][0] * image.shape[1]), int(landmarks[20][1] * image.shape[0]))]  # 새끼 손가락 끝점
                     Right_wrist = (int(landmarks[0][0] * image.shape[1]), int(landmarks[0][1] * image.shape[0]))  # 손목 좌표



                elif landmarks[20][0] > landmarks[4][0]:
                     L_fingertips = [(int(landmarks[4][0] * image.shape[1]), int(landmarks[4][1] * image.shape[0])),  # 엄지 끝점
                                        (int(landmarks[8][0] * image.shape[1]), int(landmarks[8][1] * image.shape[0])),  # 검지 끝점
                                        (int(landmarks[12][0] * image.shape[1]), int(landmarks[12][1] * image.shape[0])),  # 중지 끝점
                                        (int(landmarks[16][0] * image.shape[1]), int(landmarks[16][1] * image.shape[0])),  # 약지 끝점
                                        (int(landmarks[20][0] * image.shape[1]), int(landmarks[20][1] * image.shape[0]))]  # 새끼 손가락 끝점
                     Left_wrist = (int(landmarks[0][0] * image.shape[1]), int(landmarks[0][1] * image.shape[0]))  # 손목 좌표

                
                if prev_R_fingertip and prev_R_wrist and prev_L_fingertip and prev_L_wrist:
                    Right_fingertip_movements = np.linalg.norm(np.array(R_fingertips) - np.array(prev_R_fingertip), axis=1)
                    Right_wrist_movement = np.linalg.norm(np.array(Right_wrist) - np.array(prev_R_wrist))
                    Left_fingertip_movements = np.linalg.norm(np.array(L_fingertips) - np.array(prev_L_fingertip), axis=1)
                    Left_wrist_movement = np.linalg.norm(np.array(Left_wrist) - np.array(prev_L_wrist))

                    if any(Right_fingertip_movements > fingertip_movement_threshold) and Right_wrist_movement < wrist_movement_threshold:
                       contact_fingertips = np.where(Right_fingertip_movements > fingertip_movement_threshold)[0]
                       for fingertip in contact_fingertips:
                            if R_fingertips[fingertip][1] > prev_R_fingertip[fingertip][1]:
                                hand_label = "Right Contact"
                                if 258 < R_fingertips[fingertip][0] < 469 and 258 < R_fingertips[fingertip][1] < 282:
                                    pyautogui.press('space')
                                    break

                                elif 120 < R_fingertips[fingertip][0] < 171 and 298 < R_fingertips[fingertip][1] < 322:
                                    pyautogui.press('shiftright')
                                    break
                                
                                elif 530 < R_fingertips[fingertip][0] < 578 and 292 < R_fingertips[fingertip][1] < 317:
                                    pyautogui.press('shiftleft')
                                    break

                                elif 120 < R_fingertips[fingertip][0] < 171 and 334 < R_fingertips[fingertip][1] < 357:
                                    pyautogui.press('enter')
                                    break

                                else:
                                    for key, coord in key_coordinates.items():
                                        distance = math.dist(R_fingertips[fingertip], coord)
                                        if distance < radius:                                        
                                            pyautogui.press(key)
                                            break


                    elif any(Left_fingertip_movements > fingertip_movement_threshold) and Left_wrist_movement < wrist_movement_threshold:
                        contact_fingertips = np.where(Left_fingertip_movements > fingertip_movement_threshold)[0]
                        for fingertip in contact_fingertips:
                            if L_fingertips[fingertip][1] > prev_L_fingertip[fingertip][1]:
                                hand_label = "Left Contact"
                                if 258 < L_fingertips[fingertip][0] < 469 and 258 < L_fingertips[fingertip][1] < 282:
                                    pyautogui.press('space')
                                    break

                                elif 120 < L_fingertips[fingertip][0] < 171 and 298 < L_fingertips[fingertip][1] < 322:
                                    pyautogui.press('shiftright')
                                    break
                                
                                elif 530 < L_fingertips[fingertip][0] < 578 and 292 < L_fingertips[fingertip][1] < 317:
                                    pyautogui.press('shiftleft')
                                    break

                                elif 120 < L_fingertips[fingertip][0] < 171 and 334 < L_fingertips[fingertip][1] < 357:
                                    pyautogui.press('enter')
                                    break

                                else:
                                    for key, coord in key_coordinates.items():
                                        distance = math.dist(L_fingertips[fingertip], coord)
                                        if distance < radius:                                        
                                            pyautogui.press(key)
                                            break

                    else:
                        hand_label = " "

                elif prev_R_fingertip and prev_R_wrist:
                    Right_fingertip_movements = np.linalg.norm(np.array(R_fingertips) - np.array(prev_R_fingertip), axis=1)
                    Right_wrist_movement = np.linalg.norm(np.array(Right_wrist) - np.array(prev_R_wrist))

                    if any(Right_fingertip_movements > fingertip_movement_threshold) and Right_wrist_movement < wrist_movement_threshold:
                        contact_fingertips = np.where(Right_fingertip_movements > fingertip_movement_threshold)[0]
                        for fingertip in contact_fingertips:
                            if R_fingertips[fingertip][1] > prev_R_fingertip[fingertip][1]:
                                hand_label = "Right Contact"
                                if 258 < R_fingertips[fingertip][0] < 469 and 258 < R_fingertips[fingertip][1] < 282:
                                    pyautogui.press('space')
                                    break

                                elif 120 < R_fingertips[fingertip][0] < 171 and 298 < R_fingertips[fingertip][1] < 322:
                                    pyautogui.press('shiftright')
                                    break
                                
                                elif 530 < R_fingertips[fingertip][0] < 578 and 292 < R_fingertips[fingertip][1] < 317:
                                    pyautogui.press('shiftleft')
                                    break

                                elif 120 < R_fingertips[fingertip][0] < 171 and 334 < R_fingertips[fingertip][1] < 357:
                                    pyautogui.press('enter')
                                    break

                                else:
                                    for key, coord in key_coordinates.items():
                                        distance = math.dist(R_fingertips[fingertip], coord)
                                        if distance < radius:                                        
                                            pyautogui.press(key)
                                            break

                    else:
                        hand_label = " "
                                    
                elif prev_L_fingertip and prev_L_wrist:
                    Left_fingertip_movements = np.linalg.norm(np.array(L_fingertips) - np.array(prev_L_fingertip), axis=1)
                    Left_wrist_movement = np.linalg.norm(np.array(Left_wrist) - np.array(prev_L_wrist))

                    if any(Left_fingertip_movements > fingertip_movement_threshold) and Left_wrist_movement < wrist_movement_threshold:
                        contact_fingertips = np.where(Left_fingertip_movements > fingertip_movement_threshold)[0]
                        for fingertip in contact_fingertips:
                            if L_fingertips[fingertip][1] > prev_L_fingertip[fingertip][1]:
                                hand_label = "Left Contact"

                                if 258 < L_fingertips[fingertip][0] < 469 and 258 < L_fingertips[fingertip][1] < 282:
                                    pyautogui.press('space')
                                    break

                                elif 120 < L_fingertips[fingertip][0] < 171 and 298 < L_fingertips[fingertip][1] < 322:
                                    pyautogui.press('shiftright')
                                    break
                                
                                elif 530 < L_fingertips[fingertip][0] < 578 and 292 < L_fingertips[fingertip][1] < 317:
                                    pyautogui.press('shiftleft')
                                    break

                                elif 120 < L_fingertips[fingertip][0] < 171 and 334 < L_fingertips[fingertip][1] < 357:
                                    pyautogui.press('enter')
                                    break

                                else:
                                    for key, coord in key_coordinates.items():
                                        distance = math.dist(L_fingertips[fingertip], coord)
                                        if distance < radius:                                        
                                            pyautogui.press(key)
                                            break
                    else:
                        hand_label = " "

                else: 
                    hand_label = " "

                prev_R_fingertip = R_fingertips
                prev_L_fingertip = L_fingertips
                prev_R_wrist = Right_wrist
                prev_L_wrist = Left_wrist

                # 손 감지 결과를 이미지에 그리기
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(image, hand_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                

        # 화면에 이미지 표시
        cv2.imshow('MediaPipe Hands', image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 비디오 캡처와 윈도우를 정리하고 종료
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_hands()