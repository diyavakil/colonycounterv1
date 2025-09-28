from streamlit_drawable_canvas import st_canvas
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# --- Page setup ---
st.set_page_config(page_title="Colony Counter v1")
st.title("🧫 Colony Counter v1")

# --- Load YOLO model (CACHED) ---
@st.cache_resource
def load_yolo_model(path: str):
    """Loads the YOLO model and caches it."""
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Ensure your 'weights.pt' file is in the same directory as your app, or use a full path.
model = load_yolo_model("weights.pt")

# --- Sidebar: Correction tools ---
st.sidebar.header("Correction Tools")
mode = st.sidebar.radio("Mode", ["None", "➕ Add", "➖ Remove"])

# --- Upload image ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# A placeholder for the image data to be used later
img = None

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Check if image was successfully loaded
    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

        # --- Run YOLO inference ---
        if st.button("Run YOLO Inference"):
            # The heavy computation is wrapped here
            with st.spinner('Running inference...'):
                results = model(img)
            
            img_annotated = img.copy()

            # Draw green bounding boxes
            # Check if results are valid and have boxes
            if results and len(results) > 0 and results[0].boxes:
                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Count colonies
                colony_count = len(results[0].boxes.xyxy)
            else:
                img_annotated = img.copy() # Use unannotated image if no results
                colony_count = 0
                st.info("YOLO did not detect any colonies.")

            # Save to session state
            st.session_state["img_annotated"] = img_annotated
            st.session_state["colony_count"] = colony_count
    else:
        st.error("Could not decode the image file. Please check the file's integrity.")

## Explanation of Key Fixes

### 1. Model Caching 🚀

```python
@st.cache_resource
def load_yolo_model(path: str):
    # ... loading logic ...
model = load_yolo_model("weights.pt")
