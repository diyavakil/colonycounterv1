import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import math

# --- Configuration & Initialization ---
try:
    im = Image.open("App_Icon.jpg") 
except FileNotFoundError:
    im = "🧫"
st.set_page_config(page_title="Colony Counter (Simplified)", page_icon=im)

st.title("🔬 Colony Counter (Simplified, Single Page)")
st.markdown("---")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_yolo_model(path="weights.pt"):
    """Loads the YOLO model once and caches it."""
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Error: Could not load 'weights.pt'. Please ensure it is in the same directory. Details: {e}")
        st.stop()

model = load_yolo_model()

# --- Helper Functions ---

def cv_to_rgb_array(img_cv):
    """Converts OpenCV BGR image to RGB array for Streamlit."""
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

def annotate_image(img_cv, boxes, count, color=(0, 255, 0)):
    """Draws YOLO bounding boxes and the final count text."""
    img_annotated = img_cv.copy()
    
    # 1. Draw Boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 1)

    # 2. Draw Count Text
    text = f"Colonies: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3 
    thickness = 5
    
    img_h, img_w, _ = img_annotated.shape
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = img_w - text_size[0] - 10 
    text_y = img_h - 10 
    cv2.putText(img_annotated, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return img_annotated

# --- Session State Initialization ---
if 'yolo_run' not in st.session_state:
    st.session_state['yolo_run'] = False
if 'final_boxes' not in st.session_state:
    st.session_state['final_boxes'] = []
if 'initial_count' not in st.session_state:
    st.session_state['initial_count'] = 0
if 'img_cv_original' not in st.session_state:
    st.session_state['img_cv_original'] = None

# --- Main App Flow ---

# 1. Upload & Inference
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and convert image data
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Store the original image (needed for every re-run)
    st.session_state['img_cv_original'] = img_cv_original.copy()
    
    st.image(cv_to_rgb_array(img_cv_original), caption="Original Image", use_column_width=True)

    if st.button("Run YOLO Inference"):
        with st.spinner('Running YOLO inference...'):
            results = model(img_cv_original, verbose=False)
            yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
            initial_count = len(yolo_boxes)
        
        # Store results for the interactive step
        st.session_state['final_boxes'] = yolo_boxes
        st.session_state['initial_count'] = initial_count
        st.session_state['yolo_run'] = True
        
        st.success(f"YOLO Inference complete. Initial count: {initial_count}")
        st.experimental_rerun() # Rerun to show the interactive section

# 2. Interactive Editing (Only shown after inference)
if st.session_state['yolo_run']:
    st.markdown("---")
    st.subheader("Manual
