import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Load your trained model
with open("model.p", "rb") as f:
    model = pickle.load(f)

labels_dict = {0: 'I', 1: 'L', 2: 'Y'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# Streamlit UI
st.title("Sign Language Recognition Web App")
st.write("Show your hand signs to the webcam and see predictions in real time.")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class HandSignTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        H, W, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # Prepare data for prediction
                data_aux = []
                x_ = []
                y_ = []
                for lm in hand_landmarks.landmark:
                    x_ = [lm.x for lm in hand_landmarks.landmark]
                    y_ = [lm.y for lm in hand_landmarks.landmark]
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x)
                    data_aux.append(lm.y)

                # Predict only if correct shape
                if len(data_aux) == 42:
                    prediction = model.predict([np.asarray(data_aux)])[0]
                    predicted_character = labels_dict[int(prediction)]

                    # Bounding box
                    x1 = int(min(x_)*W) - 10
                    y1 = int(min(y_)*H) - 10
                    x2 = int(max(x_)*W) + 10
                    y2 = int(max(y_)*H) + 10

                    cv2.putText(img, predicted_character, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
                    cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)

        return img

# Start webcam streamer
webrtc_streamer(
    key="hand-sign",
    video_transformer_factory=HandSignTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)
