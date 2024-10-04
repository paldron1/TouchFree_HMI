import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
from collections import deque
from datetime import datetime
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Load the Random Forest model trained on hand landmarks
model = joblib.load('hand_landmark_random_forest_model.pkl')

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to preprocess hand landmarks for prediction
def preprocess_landmarks(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

# Function to predict hand pose based on hand landmarks
def predict_hand_pose(landmarks):
    processed_landmarks = preprocess_landmarks(landmarks)
    prediction = model.predict([processed_landmarks])
    return prediction[0]

# Logging predictions in a list
logged_predictions = []

# Class mapping
class_map = {0: 'Start Particle Counter', 1: 'Pause Particle Counter', 2: 'Stop Particle Counter'}

# Variables to track consistent detection
current_class = None
class_start_time = None
consistency_duration = 2  # 2 seconds to wait before displaying the new class
last_predictions = deque(maxlen=3)

# Streamlit app layout
st.title("PALDRON: TouchFree HMI")
st.image("img_pldrn.png", use_column_width=True)  # Background image

# Streamlit-webrtc video transformer
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.last_predictions = deque(maxlen=3)
        self.current_class = None
        self.class_start_time = None
        self.logged_predictions = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hands and get hand landmarks
        result = self.hands.process(rgb_frame)
        detected_class = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Predict hand pose based on landmarks
                predicted_class = predict_hand_pose(hand_landmarks.landmark)
                detected_class = class_map[predicted_class]

                # Get bounding box coordinates around the hand
                h, w, _ = img.shape
                x_min, x_max, y_min, y_max = w, 0, h, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, x_max = min(x, x_min), max(x, x_max)
                    y_min, y_max = min(y, y_min), max(y, y_max)

                # Draw bounding box around the hand
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Display the last 3 classifications to the far left in red
                for i, pred in enumerate(reversed(self.last_predictions)):
                    cv2.putText(img, pred, (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Check if the detected class is consistent for 2 seconds
        if detected_class:
            if detected_class == self.current_class:
                elapsed_time = time.time() - self.class_start_time
                if elapsed_time >= consistency_duration:
                    if len(self.last_predictions) == 0 or self.last_predictions[-1] != self.current_class:
                        self.last_predictions.append(self.current_class)
                        self.logged_predictions.append({
                            "time_stamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "command": self.current_class
                        })
            else:
                self.current_class = detected_class
                self.class_start_time = time.time()

        return img
import asyncio

# Ensure the asyncio event loop is properly initialized
if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())

# Define streamlit_webrtc component for video streaming
webrtc_streamer(key="hand_pose_detection", mode=WebRtcMode.SENDRECV, video_processor_factory=VideoTransformer)


# Print results to a spreadsheet when Print button is clicked
if st.button("Print Results") and logged_predictions:
    df = pd.DataFrame(logged_predictions)
    file_name = f'logged_predictions_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    df.to_csv(file_name, index=False)
    st.success(f"Results saved as {file_name}")
    st.dataframe(df)

# Download CSV button logic
if st.button("Download CSV") and logged_predictions:
    df = pd.DataFrame(logged_predictions)
    file_name = f'logged_predictions_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    csv = df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name=file_name, mime='text/csv')
