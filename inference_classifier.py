import cv2
import mediapipe as mp
from train_classifier import model
import pickle
import numpy as np

model_dict = pickle.load(open('model.p', 'rb'))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3,
                       min_tracking_confidence=0.3)

cap = cv2.VideoCapture(0)

labels_dict = {0:'I', 1:'L', 2:'Y'}
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W , _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        for hand_landmarks in results.multi_hand_landmarks:  # for each hand_landmarks, x = width, y = height, z = depth ie how close or far from webcam

            for i in range(len(hand_landmarks.landmark)):  # 21 sets of x, y and z value
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])[0]
            predicted_character = labels_dict[int(prediction)]

        x1 = int(min(x_)*W) -10
        y1 = int(min(y_)*H) -10
        x2 = int(max(x_)*W) + 10
        y2 = int(max(y_)*H) + 10

        print(predicted_character)
        cv2.putText(frame, predicted_character, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
