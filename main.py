import mediapipe as mp
import cv2
import numpy as np
import os
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data_dir = '/Users/nisumlimbu/PycharmProjects/Sign_language_detection/data'

data = []
labels = []

for dir_ in os.listdir(data_dir):
    if not os.path.isdir(os.path.join(data_dir, dir_)):
        continue # this skips non directory files like DS_Store
    for img_path in os.listdir(os.path.join(data_dir, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(data_dir, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)  # process each image of hand
        if results.multi_hand_landmarks:  # reutrns multi_hand_landmarks of all hands = x,y,z coordinates for 21 points of each hand
            for hand_landmarks in results.multi_hand_landmarks:  #for each hand_landmarks, x = width, y = height, z = depth ie how close or far from webcam

                for i in range(len(hand_landmarks.landmark)):  # 21 sets of x, y and z value
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            labels.append(dir_)
f = open('data.pkl', 'wb')
pickle.dump({'data':data, 'labels':labels}, f)
f.close()


