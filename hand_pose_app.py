import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
from collections import deque
from datetime import datetime
import time

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

# Increase video size by 5% (e.g., if default width is 640px, it will be 672px)
image_width_percent = 1.05  # 5% increase in size

# Video capture placeholder above the buttons
video_placeholder = st.empty()

# Define a horizontal layout for the buttons at the bottom of the video
button_col1, button_col2, button_col3, button_col4 = st.columns([1, 1, 1, 1])

# Initialize variables
cap = None
running = False

# Start video capture when Start button is clicked
with button_col1:
    start_button = st.button("Start")

# Stop video capture when Stop button is clicked
with button_col2:
    stop_button = st.button("Stop")

# Print results to a spreadsheet when Print button is clicked
with button_col3:
    print_button = st.button("Print Results")

# Download CSV button
with button_col4:
    download_button = st.button("Download CSV")

# Start video capture
if start_button:
    running = True
    cap = cv2.VideoCapture(1)

# Stop video capture when Stop button is clicked
if stop_button and cap:
    cap.release()
    running = False

# Capture and display video in real-time
if running and cap is not None:
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands and get hand landmarks
            result = hands.process(rgb_frame)

            detected_class = None

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw hand landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Predict hand pose based on landmarks
                    predicted_class = predict_hand_pose(hand_landmarks.landmark)
                    detected_class = class_map[predicted_class]

                    # Get bounding box coordinates around the hand
                    h, w, _ = frame.shape
                    x_min, x_max, y_min, y_max = w, 0, h, 0
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min, x_max = min(x, x_min), max(x, x_max)
                        y_min, y_max = min(y, y_min), max(y, y_max)

                    # Draw bounding box around the hand
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Display the last 3 classifications to the far left in red
                    for i, pred in enumerate(reversed(last_predictions)):
                        cv2.putText(frame, pred, (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Check if the detected class is consistent for 2 seconds
            if detected_class:
                if detected_class == current_class:
                    elapsed_time = time.time() - class_start_time
                    if elapsed_time >= consistency_duration:
                        if len(last_predictions) == 0 or last_predictions[-1] != current_class:
                            last_predictions.append(current_class)
                            logged_predictions.append({
                                "time_stamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "command": current_class
                            })
                else:
                    current_class = detected_class
                    class_start_time = time.time()

            # Display the frame with landmarks, bounding box, and predictions
            frame_width = int(frame.shape[1] * image_width_percent)
            frame_height = int(frame.shape[0] * image_width_percent)
            resized_frame = cv2.resize(frame, (frame_width, frame_height))
            video_placeholder.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), channels="RGB")

# Print results to a spreadsheet when Print button is clicked
if print_button and logged_predictions:
    df = pd.DataFrame(logged_predictions)
    file_name = f'logged_predictions_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    df.to_csv(file_name, index=False)
    st.success(f"Results saved as {file_name}")
    st.dataframe(df)

# Download CSV button logic
if download_button and logged_predictions:
    df = pd.DataFrame(logged_predictions)
    file_name = f'logged_predictions_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    csv = df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name=file_name, mime='text/csv')



