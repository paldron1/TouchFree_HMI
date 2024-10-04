import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

# Load your trained Random Forest model
import joblib
model = joblib.load('hand_landmark_random_forest_model.pkl')

# Mediapipe initialization for hand landmarks
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to preprocess hand landmarks for prediction
def preprocess_landmarks(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

# Helper function to decode base64 image data to OpenCV format
def decode_base64_to_image(image_base64):
    image_bytes = base64.b64decode(image_base64.split(',')[1])
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

# Function to run hand pose detection
def detect_hand_pose(image):
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_image)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                processed_landmarks = preprocess_landmarks(hand_landmarks.landmark)
                predicted_class = model.predict([processed_landmarks])[0]
                class_map = {0: 'Start Particle Counter', 1: 'Pause Particle Counter', 2: 'Stop Particle Counter'}
                return image, class_map[predicted_class]
    return image, None

# JavaScript for accessing webcam and capturing images
webcam_html = """
    <script>
    let video = document.createElement('video');
    let canvas = document.createElement('canvas');
    video.style.display = 'none';
    canvas.style.display = 'none';
    document.body.appendChild(video);
    document.body.appendChild(canvas);
    
    async function startWebcam() {
        let stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
        return stream;
    }

    async function captureImage() {
        if (!video.srcObject) {
            await startWebcam();
        }
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        let context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        let dataUrl = canvas.toDataURL('image/jpeg');
        return dataUrl;
    }

    window.captureWebcamImage = async function() {
        return await captureImage();
    }
    </script>
"""


# Streamlit app layout
st.title("PALDRON: TouchFree HMI")
st.image("img_pldrn.png", use_column_width=True)  # Background image


# Embed webcam HTML and JS into Streamlit
st.components.v1.html(webcam_html, height=0)

# Button to capture image from webcam
if st.button('Capture Image from Webcam'):
    image_data_url = st.js.eval("window.captureWebcamImage()")
    
    if image_data_url:
        # Convert base64 image data to OpenCV image
        image = decode_base64_to_image(image_data_url)

        # Perform hand pose detection on the captured image
        image, detected_class = detect_hand_pose(image)

        # Display the processed image
        st.image(image, caption=f"Captured Image - Detected Pose: {detected_class}", use_column_width=True)

        # Log the detected class with a timestamp if a pose is detected
        if detected_class:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"Detected Pose: {detected_class} at {timestamp}")

# Download button to save image data (Optional)
if st.button('Download Captured Image'):
    if image_data_url:
        img = decode_base64_to_image(image_data_url)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        b64 = base64.b64encode(byte_im).decode()
        href = f'<a href="data:file/jpg;base64,{b64}" download="captured_image.jpg">Download captured image</a>'
        st.markdown(href, unsafe_allow_html=True)
