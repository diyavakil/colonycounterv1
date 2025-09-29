import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# --- Configuration & Initialization ---
try:
    im = Image.open("App_Icon.jpg") 
except FileNotFoundError:
    im = "🧫"
st.set_page_config(page_title="Colony Counter", page_icon=im)

st.title("🧫 Colony Counter: Upload & Inference")
st.markdown("---")

# --- KEY FIX: Use st.cache_resource for the Model ---
@st.cache_resource
def load_yolo_model(path="weights.pt"):
    """Loads the YOLO model once and caches it."""
    try:
        # Load weights (assuming weights.pt is in the same directory)
        return YOLO(path)
    except Exception as e:
        st.error(f"Error: Could not load 'weights.pt'. Please ensure it is in the same directory. Details: {e}")
        st.stop()

model = load_yolo_model()
# ----------------------------------------------------
    
# Initialize list to hold processed images and data for the session
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = []
    
# --- Helper Function for Annotation (NO COUNT TEXT) ---
def annotate_image_no_text(img_cv, boxes):
    """Draws YOLO bounding boxes on the image without the count text."""
    img_annotated = img_cv.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Draw green bounding boxes (BGR: 0, 255, 0)
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return img_annotated

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # Read and convert to OpenCV (BGR)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  
    
    # Store original image for later comparison/use
    st.session_state['current_original_cv'] = img_cv.copy()
    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption=f"Original Image: {uploaded_file.name}", use_column_width=True)

    # --- YOLO Inference and Storage ---
    if st.button("Run YOLO Inference and Go to Editor"):
        
        with st.spinner(f'Running YOLO inference on {uploaded_file.name}...'):
            # Use the globally cached model object
            results = model(img_cv, verbose=False)
            yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
            initial_count = len(yolo_boxes)
            
            # 1. Create the base annotated image (YOLO boxes ONLY, NO COUNT TEXT)
            img_annotated_cv = annotate_image_no_text(img_cv, yolo_boxes)

            # 2. Store ALL data for the session
            image_id = uploaded_file.name
            
            # Store the data required for the next page
            st.session_state['processed_data'].append({
                'id': image_id,
                'initial_count': initial_count,
                'annotated_cv': img_annotated_cv, # Image with YOLO boxes only
                'original_cv': img_cv,           # Original clean image
                'yolo_boxes': yolo_boxes         # Original box coordinates
            })
            
            # Set the ID of the image we just processed for auto-selection on the next page
            st.session_state['current_edit_id'] = image_id
            
            st.success(f"Inference complete. Colonies found: {initial_count}")
            
            # --- Automatic Navigation ---
            st.switch_page("pages/1_Colony_Editor.py")
