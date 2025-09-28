import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- Streamlit Caching ---

# 1. Cache the YOLO model: This prevents reloading the weights.pt file on every single interaction.
@st.cache_resource
def load_yolo_model(path):
    """Loads the YOLO model and caches it."""
    return YOLO(path)

# 2. Cache the initial YOLO inference results: This prevents re-running the heavy inference
# and initial drawing on every click.

# THE FIX IS HERE: Renamed 'model' to '_model' to exclude it from the cache hash calculation.
@st.cache_data
def run_initial_inference(img, _model):
    """Runs YOLO inference and returns the initial annotated image (CV2) and count."""
    results = _model(img) # Use _model inside the function
    img_annotated_cv2 = img.copy()
    
    # Draw YOLO bboxes (green)
    yolo_bboxes = results[0].boxes.xyxy.cpu().numpy()
    for box in yolo_bboxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_annotated_cv2, (x1, y1), (x2, y2), (0, 255, 0), 1)

    yolo_count = len(yolo_bboxes)
    
    # Convert BGR (CV2) to RGB (PIL) for the Canvas background
    img_annotated_rgb = cv2.cvtColor(img_annotated_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_annotated_rgb)

    return img_annotated_cv2, img_pil, yolo_count


# --- Streamlit Session State Initialization ---
# Initialize necessary state variables right at the start
if 'inference_run' not in st.session_state:
    st.session_state['inference_run'] = False
if 'img_annotated_cv2' not in st.session_state:
    st.session_state['img_annotated_cv2'] = None
if 'img_annotated_pil' not in st.session_state:
    st.session_state['img_annotated_pil'] = None
if 'yolo_count' not in st.session_state:
    st.session_state['yolo_count'] = 0
if 'img_original_cv' not in st.session_state:
    st.session_state['img_original_cv'] = None
if 'uploaded_file_id' not in st.session_state:
    st.session_state['uploaded_file_id'] = None
if 'canvas_json_data' not in st.session_state:
    st.session_state['canvas_json_data'] = None

# custom title/icon
im = Image.open("App_Icon.jpg") # cute squid
st.set_page_config(page_title="Colony Counter v1", page_icon=im)

# header
st.title("🧫 Colony Counter v1")

# Load model (using cache)
model = load_yolo_model("weights.pt")


# --- Callback Function for Inference Button ---
def run_inference_callback(img, model):
    """Handles the heavy lifting (YOLO inference) and stores results in state."""
    
    # Run the initial inference using the cached function
    # THE FIX IS HERE: The function call is correct because the cached function now accepts '_model'
    img_cv2, img_pil, yolo_count = run_initial_inference(img, model) 
    
    # Store results in session state
    st.session_state['yolo_count'] = yolo_count
    st.session_state['img_annotated_cv2'] = img_cv2 # CV2 version for final drawing/saving
    st.session_state['img_annotated_pil'] = img_pil # PIL version for canvas background
    st.session_state['inference_run'] = True
    st.session_state['canvas_json_data'] = None # Clear any previous manual drawing


# --- Function to finalize image with manual points and count for saving ---
def finalize_image_for_download(img_base_cv2, manual_points_data, yolo_count):
    """Draws manual points and the final count text onto the CV2 image array."""
    
    final_img = img_base_cv2.copy()
    
    # 1. Draw manual points (red dots) from canvas data
    manual
