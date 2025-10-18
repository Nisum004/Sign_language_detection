# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import cv2
import av
import numpy as np

# -------------------
# Mediapipe setup
# -------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Hand Sign Detection", layout="wide")
st.title("🤚 Hand Sign Detection App")

st.write("This app detects your hand landmarks in real-time using your webcam.")

# Optional: add instructions
st.markdown("""
- Make sure your webcam is on.
- Show your hand clearly in front of the camera.
- The detected landmarks will be drawn on your hand.
""")


# -------------------
# Video callback for streamlit-webrtc
# -------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")  # OpenCV image

    # Flip for mirror view
    img = cv2.flip(img, 1)

    # Convert BGR to RGB for Mediapipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------
# Run webcam with webrtc
# -------------------
webrtc_streamer(
    key="hand-sign-detection",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)
