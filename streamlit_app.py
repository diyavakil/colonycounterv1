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

# Ensure your 'weights.pt' file is accessible
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
                st.info(f"YOLO Detected Colonies: {colony_count}") # Provide initial count
            else:
                img_annotated = img.copy() # Use unannotated image if no results
                colony_count = 0
                st.info("YOLO did not detect any colonies.")

            # Save to session state
            st.session_state["img_annotated"] = img_annotated
            st.session_state["colony_count"] = colony_count
    else:
        st.error("Could not decode the image file. Please check the file's integrity.")

# --------------------------------------------------------------------------------------
# --- Display annotated image and correction canvas (FINAL PART) ---
# --------------------------------------------------------------------------------------

# Only display this section after inference has run and state variables are populated
if "img_annotated" in st.session_state:
    img_annotated = st.session_state["img_annotated"]
    colony_count = st.session_state["colony_count"]

    st.subheader("📊 Annotated Image & Corrections")
    st.write(f"**Initial YOLO Count:** {colony_count}")

    # --- Canvas ---
    # Convert OpenCV BGR image to PIL RGB for st_canvas background
    background_img_pil = Image.fromarray(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
    
    # Ensure drawing_mode is 'point' for adding/removing, and 'none' otherwise
    current_drawing_mode = "point" if mode in ["➕ Add", "➖ Remove"] else "none"

    canvas_result = st_canvas(
        fill_color="rgba(0,255,0,0.7)" if mode == "➕ Add" else "rgba(255,0,0,0.7)",
        stroke_width=5,
        stroke_color="green" if mode == "➕ Add" else "red",
        background_image=background_img_pil,
        update_streamlit=True,
        height=img_annotated.shape[0],
        width=img_annotated.shape[1],
        drawing_mode=current_drawing_mode,
        key="correction_canvas"
    )

    # --- Count corrections safely ---
    added = 0
    removed_drawn = 0

    if canvas_result.json_data and "objects" in canvas_result.json_data:
        # Filter objects to count only valid "points" drawn by the user
        points_drawn = [obj for obj in canvas_result.json_data["objects"] if obj.get("type") == "point"]
        
        if mode == "➕ Add":
            added = len(points_drawn)
        elif mode == "➖ Remove":
            # Points drawn in "Remove" mode are used to indicate removals
            removed_drawn = len(points_drawn)

    # --- Corrected count ---
    # We ensure the count doesn't go below zero
    corrected_count = max(0, colony_count + added - removed_drawn)

    st.success(f"✅ **Corrected Colony Count:** {corrected_count}")
