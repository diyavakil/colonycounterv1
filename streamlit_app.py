import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import json
import base64
from io import BytesIO

# --- Configuration & Initialization ---

# custom title/icon
im = Image.open("App_Icon.jpg") # cute squid
st.set_page_config(page_title="Colony Counter v2 (Editable)", page_icon=im)

# header
st.title("🧫 Colony Counter v2 (Editable)")

# Load weights (per user request, assuming it works)
model = YOLO("weights.pt") 

# Function to convert YOLO boxes to a simple JSON list of centers for the canvas
def yolo_boxes_to_initial_dots_json(boxes, stroke_color, radius=5):
    """Converts YOLO bounding boxes to 'circle' objects for st_canvas using the center of the box."""
    canvas_objects = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(float, box)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        canvas_objects.append({
            "type": "circle",
            "left": center_x,
            "top": center_y,
            "radius": radius,
            "fill": stroke_color,
            "stroke": stroke_color,
            "strokeWidth": 1,
            "originX": "center",
            "originY": "center",
            "point_type": "yolo_detection" # Custom tag for original detections
        })
    
    return {"objects": canvas_objects, "background": ""}

# --- Image Upload ---

# upload img
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    
    # 1. Convert uploaded file to OpenCV img and display original
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  
    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

    if st.button("Run YOLO Inference"):
        
        # 2. Run YOLO Inference and prepare initial state
        results = model(img_cv, verbose=False)
        yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
        initial_colony_count = len(yolo_boxes)
        
        # Convert the original image to PIL Image for st_canvas background
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Prepare initial canvas drawing data from YOLO results
        initial_drawing = yolo_boxes_to_initial_dots_json(yolo_boxes, "#00FF00") # Green for initial detections
        
        # Store essential data in session state
        st.session_state['yolo_boxes'] = yolo_boxes # Keep boxes for potential later use
        st.session_state['initial_drawing_json'] = initial_drawing
        st.session_state['img_pil'] = img_pil
        st.session_state['img_cv'] = img_cv # Keep original CV image
        st.session_state['image_dims'] = img_cv.shape[:2] # (height, width)

# --- Interactive Editing Canvas (Runs after inference button is clicked) ---

if 'img_pil' in st.session_state:
    st.markdown("---")
    st.subheader("Interactive Colony Editor")
    
    # 3. Setup Drawing Controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        draw_mode = st.radio(
            "Select action:",
            ("Add Colony", "Remove Colony"),
            key="draw_mode"
        )
        
        if draw_mode == "Add Colony":
            stroke_color = "#00FF00"  # Green
        else: # Remove Colony
            stroke_color = "#FF0000"  # Red

        st.info(f"Click on the image to {draw_mode.lower()} (Dot color: {stroke_color})")
        
    # 4. Display the Canvas
    with col2:
        # Use canvas to handle interactive drawing
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0.0)", # No fill
            stroke_width=5, 
            stroke_color=stroke_color,
            background_image=st.session_state['img_pil'],
            initial_drawing=st.session_state['initial_drawing_json'],
            update_streamlit=True,
            height=st.session_state['image_dims'][0], # Use original height
            width=st.session_state['image_dims'][1],  # Use original width
            drawing_mode="point",
            point_display_radius=5,
            display_toolbar=False,
            key="colony_editor_canvas",
        )

    # 5. Process Canvas Results and Display Metrics
    if canvas_result.json_data is not None:
        all_objects = canvas_result.json_data.get("objects", [])
        
        # Recalculate the count
        # 1. Count YOLO detections (points with custom tag)
        yolo_points_count = sum(1 for obj in all_objects if obj.get('point_type') == 'yolo_detection')
        
        # 2. Count user-added points (new green points)
        # Note: We check for 'circle' type since only circles are created in point mode.
        added_points_count = sum(1 for obj in all_objects if obj.get('stroke') == '#00FF00' and obj.get('point_type') != 'yolo_detection' and obj.get('type') == 'circle')
        
        # 3. Count user-removed points (new red points)
        removed_points_count = sum(1 for obj in all_objects if obj.get('stroke') == '#FF0000' and obj.get('type') == 'circle')
        
        final_count = yolo_points_count + added_points_count - removed_points_count
        
        st.markdown("---")
        st.subheader("Colony Count Summary")
        
        # Display metrics
        colony_count_initial = len(st.session_state.get('yolo_boxes', []))
        st.metric(label="Initial YOLO Count", value=colony_count_initial)
        st.metric(label="User Added (Green dots)", value=added_points_count)
        st.metric(label="User Marked for Removal (Red dots)", value=removed_points_count)
        st.metric(label="**Final Colony Count**", value=final_count, delta=final_count - colony_count_initial, delta_color="normal")
        
        st.markdown("---")

        # 6. Recreate the final annotated OpenCV image (re-using original text drawing logic)
        
        # Get the final image data from the canvas result
        img_final_np = canvas_result.image_data
        
        # Convert RGBA numpy array to BGR OpenCV format
        img_annotated = cv2.cvtColor(img_final_np.astype('uint8'), cv2.COLOR_RGBA2BGR)

        # Add count in bottom-right corner using original OpenCV logic
        text = f"Colonies: {final_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3 # text size
        thickness = 5 # text thickness
        
        # Get image dimensions
        img_h, img_w, _ = img_annotated.shape

        # Calculate text position
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = img_w - text_size[0] - 10  # 10 px from right
        text_y = img_h - 10  # 10 px from bottom
        cv2.putText(img_annotated, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness) # Green text (BGR: 0, 255, 0)
        
        # Display the final image
        st.image(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB), caption="Final Annotated Image", use_column_width=True)


        # 7. Save and Download functionality (re-using original logic)
        save_path = "annotated_streamlit.jpg"
        cv2.imwrite(save_path, img_annotated)
        st.success(f"Annotated image saved as {save_path}")
        
        # Read the saved image file for download button
        with open(save_path, "rb") as file:
            st.download_button(
                label="Download Final Annotated Image",
                data=file.read(),
                file_name="annotated_image.jpg",
                mime="image/jpeg"
            )
