import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- Streamlit Session State Initialization (FIXED) ---
# Initialize ALL necessary state variables at the absolute top for robustness
if 'yolo_state_initialized' not in st.session_state:
    st.session_state.update({
        'inference_run': False,
        'img_annotated_cv2': None,
        'img_annotated_pil': None,
        'yolo_count': 0,
        'img_original_cv': None,
        'uploaded_file_id': None,
        'canvas_json_data': None,
        'yolo_state_initialized': True # Flag to ensure this only runs once
    })

# custom title/icon
# NOTE: Ensure 'App_Icon.jpg' exists or this line will cause an error
try:
    im = Image.open("App_Icon.jpg") 
    st.set_page_config(page_title="Colony Counter v1", page_icon=im)
except FileNotFoundError:
    st.set_page_config(page_title="Colony Counter v1", page_icon="🧫")

# header
st.title("🧫 Colony Counter v1")

# --- Streamlit Caching ---

# 1. Cache the YOLO model
@st.cache_resource
def load_yolo_model(path):
    """Loads the YOLO model and caches it."""
    # NOTE: Ensure 'weights.pt' exists or this line will cause an error
    try:
        return YOLO(path)
    except FileNotFoundError:
        st.error(f"Error: Model file '{path}' not found. Please place it in the application directory.")
        return None

# 2. Cache the initial YOLO inference results (Fixes UnhashableParamError)
@st.cache_data
def run_initial_inference(img, _model):
    """Runs YOLO inference and returns the initial annotated image (CV2) and count."""
    
    # Check if model loaded successfully before running inference
    if _model is None:
        return None, None, 0

    results = _model(img) 
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


# --- Callback Function for Inference Button ---
def run_inference_callback(img, model):
    """Handles the heavy lifting (YOLO inference) and stores results in state."""
    
    # Run the initial inference using the cached function
    img_cv2, img_pil, yolo_count = run_initial_inference(img, model) 
    
    if img_cv2 is not None:
        # Store results in session state
        st.session_state['yolo_count'] = yolo_count
        st.session_state['img_annotated_cv2'] = img_cv2 
        st.session_state['img_annotated_pil'] = img_pil 
        st.session_state['inference_run'] = True
        st.session_state['canvas_json_data'] = None 


# --- Function to finalize image with manual points and count for saving ---
def finalize_image_for_download(img_base_cv2, manual_points_data, yolo_count):
    """Draws manual points and the final count text onto the CV2 image array."""
    
    final_img = img_base_cv2.copy()
    
    # 1. Draw manual points (red dots) from canvas data
    manual_count = 0
    if manual_points_data and 'objects' in manual_points_data:
        for obj in manual_points_data['objects']:
            if obj.get('type') == 'circle':
                # Approximate the center of the circle drawn by the user
                center_x = int(obj['left'] + obj['width'] / 2)
                center_y = int(obj['top'] + obj['height'] / 2)
                
                # Draw a small red circle
                cv2.circle(final_img, (center_x, center_y), 5, (0, 0, 255), -1) 
                manual_count += 1
                
    # 2. Calculate and draw total count
    total_count = yolo_count + manual_count
    
    text = f"Total Colonies: {total_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5 
    thickness = 3 
    
    # Get text size and position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = final_img.shape[1] - text_size[0] -
