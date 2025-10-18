import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.p', 'rb'))

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # detect up to 2 hands
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# Webcam
cap = cv2.VideoCapture(0)

# Label mapping
labels_dict = {0: 'I', 1: 'L', 2: 'Y'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Loop through each detected hand
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Extract x and y coordinates for this hand
            data_aux = []
            x_coords = []
            y_coords = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x)
                data_aux.append(lm.y)
                x_coords.append(lm.x)
                y_coords.append(lm.y)

            # Predict for this hand
            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])[0]
                predicted_character = labels_dict[int(prediction)]

                # Bounding box for this hand
                x1 = int(min(x_coords) * W) - 10
                y1 = int(min(y_coords) * H) - 10
                x2 = int(max(x_coords) * W) + 10
                y2 = int(max(y_coords) * H) + 10

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

                print(f"Hand {hand_idx + 1}: {predicted_character}")

    cv2.imshow('Hand Sign Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
