from streamlit_drawable_canvas import st_canvas
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# --- Page setup ---
st.set_page_config(page_title="Colony Counter v1")
st.title("🧫 Colony Counter v1")

# --- Load YOLO model ---
model = YOLO("weights.pt")

# --- Sidebar: Correction tools ---
st.sidebar.header("Correction Tools")
mode = st.sidebar.radio("Mode", ["None", "➕ Add", "➖ Remove"])

# --- Upload image ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

    # --- Run YOLO inference ---
    if st.button("Run YOLO Inference"):
        results = model(img)
        img_annotated = img.copy()

        # Draw green bounding boxes
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Count colonies
        colony_count = len(results[0].boxes.xyxy)

        # Save to session state
        st.session_state["img_annotated"] = img_annotated
        st.session_state["colony_count"] = colony_count

# --- Display annotated image and correction canvas ---
if "img_annotated" in st.session_state:
    img_annotated = st.session_state["img_annotated"]
    colony_count = st.session_state["colony_count"]

    st.subheader("📊 Annotated Image & Corrections")

    # --- Canvas ---
    canvas_result = st_canvas(
        fill_color="rgba(0,255,0,0.7)" if mode == "➕ Add" else "rgba(255,0,0,0.7)",
        stroke_width=5,
        stroke_color="green" if mode == "➕ Add" else "red",
        background_image=Image.fromarray(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)),
        update_streamlit=True,
        height=img_annotated.shape[0],
        width=img_annotated.shape[1],
        drawing_mode="point" if mode in ["➕ Add", "➖ Remove"] else "none",
        key="correction_canvas"
    )

    # --- Count corrections safely ---
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        added = len(canvas_result.json_data["objects"]) if mode == "➕ Add" else 0
        removed_drawn = len(canvas_result.json_data["objects"]) if mode == "➖ Remove" else 0
    else:
        added = 0
        removed_drawn = 0

    # --- Corrected count ---
    corrected_count = colony_count + added - removed_drawn
    st.success(f"✅ Corrected Colony Count: {corrected_count}")
