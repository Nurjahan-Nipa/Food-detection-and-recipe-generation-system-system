import streamlit as st
from PIL import Image
import os
import cv2
from inference import get_model
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr

# Create necessary folders
required_dirs = ["data", "data/sample", "data/uploads"]
for directory in required_dirs:
    os.makedirs(directory, exist_ok=True)

# change to OUR TRAINED MODEL!!
# I have used yolov8n-640 just for demo purpose!!
model = get_model(model_id="yolov8n-640")

# 🔍 Detection function
def detect_ingredients(image_path):
    image = load_image_bgr(image_path)
    response = model.infer(image)

    # Handle older Roboflow SDK structure
    if isinstance(response, list):  
        response = response[0]
    detections = sv.Detections.from_inference(response)
    labels = []
    if isinstance(response, dict):  
        labels = list({pred["class"] for pred in response["predictions"]})
    else:  
        labels = list({pred.class_name for pred in response.predictions})
    return labels

# Streamlit UI
st.set_page_config(page_title="CookVision", layout="centered")
st.title("🍳 CookVision: AI Cooking Assistant")
st.markdown("Upload a food image to find the ingredients used.")

uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_path = os.path.join("data/uploads", uploaded_file.name)
    # Save file
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_container_width=True)  

    # Detect ingredients
    with st.spinner("Detecting ingredients..."):
        detected_ingredients = detect_ingredients(image_path)

    # Show result
    st.subheader("🧾🔍 Detected Ingredients")
    if detected_ingredients:
        st.success(", ".join(detected_ingredients))
    else:
        st.warning("🔍 No ingredients detected.")
