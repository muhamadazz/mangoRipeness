import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from PIL import Image

# Initialize the model
model = YOLO("model.pt")

# Check if CUDA is available and move the model to GPU if it is
if torch.cuda.is_available():
    model.to('cuda')

def process_image(uploaded_image):
    # Convert PIL Image to an OpenCV image
    image = np.array(uploaded_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize the image to a smaller size for faster processing
    image_resized = cv2.resize(image, (640, 480))

    # Perform prediction
    results = model.predict(image_resized)

    # Annotate the image
    for result in results:
        for bbox, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, bbox)
            label = result.names[int(cls)]
            confidence = float(conf)
            cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_resized, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert back to RGB for display in Streamlit
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    return image_resized

st.title("Face Expression Recognition")
st.write("Upload an image to detect face expressions")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to a PIL image
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Detecting face expressions...")

    # Process the image and display the result
    processed_image = process_image(image)
    st.image(processed_image, caption='Processed Image', use_column_width=True)
