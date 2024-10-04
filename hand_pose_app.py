import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
from collections import deque
from datetime import datetime
import time
import streamlit.components.v1 as components
import base64

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

# JavaScript code for webcam access
webcam_html = """
<div style="text-align: center;">
    <video id="video" width="640" height="480" autoplay></video>
    <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
    <button id="capture">Capture Frame</button>
</div>

<script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureButton = document.getElementById('capture');

    navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
        video.srcObject = stream;
    });

    captureButton.addEventListener('click', function() {
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, 640, 480);
        const dataUrl = canvas.toDataURL('image/png');
        window.parent.postMessage(dataUrl, '*');
    });
</script>
"""

# Include the JavaScript in the Streamlit app
components.html(webcam_html)

# Capture the image when the JavaScript sends it back
image_data = st.experimental_get_query_params().get('dataUrl')
if image_data:
    image_data = image_data[0].split(',')[1]  # Strip data URL metadata
    image_bytes = base64.b64decode(image_data)

    # Convert the image to a format that OpenCV can process
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Display the captured image
    st.image(img, caption="Captured Frame", use_column_width=True)

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect hands and get hand landmarks
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        result = hands.process(rgb_frame)
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
                for i, pred in enumerate(reversed(last_predictions)):
                    cv2.putText(img, pred, (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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

    # Display updated image with hand landmarks and bounding box
    st.image(img, caption="Processed Frame with Hand Landmarks", use_column_width=True)

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
