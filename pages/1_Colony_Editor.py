import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from io import BytesIO
import base64

# --- Configuration & Initialization ---
st.set_page_config(page_title="Colony Editor")
st.title("✂️ Colony Editor (Manual Adjustments)")
st.markdown("---")

# --- Helper Functions ---

def cv_to_pil_rgb(img_cv):
    """Converts OpenCV BGR image to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# FIX: Ensure a default format for Base64, and explicitly convert to RGB
def create_initial_drawing_json(img_pil):
    """Creates a transparent drawing JSON with the image embedded in the background (Base64)."""
    
    buffered = BytesIO()
    
    # Ensure image is RGB before saving to avoid PIL warnings/errors
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
        
    # Use JPEG for smaller size and faster loading in the browser, if possible
    # This line was where the error traceback pointed, ensuring it is correct here
    img_pil.save(buffered, format="JPEG", quality=85) 
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "objects": [], 
        "background": f"data:image/jpeg;base64,{img_str}"
    }

def hex_to_bgr(h):
    """Converts a hex color code to a BGR tuple for OpenCV."""
    return tuple(int(h[i:i+2], 16) for i in (4, 2, 0)) # BGR order


# --- Main Logic ---

if 'processed_data' not in st.session_state or not st.session_state['processed_data']:
    st.warning("No processed images found. Please upload an image and run inference first.")
    if st.button("Go to Upload Page"):
        st.switch_page("streamlit_app.py")
    st.stop()

# 1. Image Selection Sidebar
st.sidebar.subheader("Processed Images")
data_list = st.session_state['processed_data']
image_options = {d['id']: d for d in data_list}

# Auto-select the last processed image, if available
default_index = 0
if 'current_edit_id' in st.session_state and st.session_state['current_edit_id'] in image_options:
    default_index = list(image_options.keys()).index(st.session_state['current_edit_id'])

selected_id = st.sidebar.selectbox(
    "Select image to edit:",
    options=list(image_options.keys()),
    index=default_index,
    key='selected_image_id'
)

current_data = image_options[selected_id]
initial_count = current_data['initial_count']
img_cv_base = current_data['annotated_cv']
img_pil_base = cv_to_pil_rgb(img_cv_base)
st.sidebar.markdown(f"**YOLO Initial Count:** {initial_count}")

# 2. Setup Canvas Controls

# FIX: Initialize draw_mode and stroke_color outside the columns to guarantee existence
# Initialize state variables
if 'draw_mode' not in st.session_state:
    st.session_state['draw_mode'] = "Add Colony"

draw_mode = st.session_state['draw_mode']
stroke_color = "#00FF00" if draw_mode == "Add Colony" else "#FF0000"


st.subheader(f"Editing: {selected_id}")

col1, col2 = st.columns([1, 2])

with col1:
    # Use a fresh st.radio call to update the session state variable
    # This ensures draw_mode is always defined before we use it to calculate stroke_color
    draw_mode = st.radio(
        "Select action:",
        ("Add Colony", "Remove Colony"),
        key="draw_mode", # Uses the session state key
    )
    
    # Recalculate stroke_color AFTER the radio button has been interacted with
    stroke_color = "#00FF00" if draw_mode == "Add Colony" else "#FF0000"
    
    # This line now runs safely because stroke_color is guaranteed to exist
    st.info(f"Dot color: {stroke_color}")
    
# 3. Display the Canvas
with col2:
    # Use the annotated image as the background
    initial_drawing_json = create_initial_drawing_json(img_pil_base)

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0.0)",
        stroke_width=5, 
        stroke_color=stroke_color,
        initial_drawing=initial_drawing_json, 
        update_streamlit=True,
        height=img_cv_base.shape[0], 
        width=img_cv_base.shape[1],
        drawing_mode="point",
        point_display_radius=5,
        display_toolbar=False,
        key=f"editor_canvas_{selected_id}", 
    )

# ... (Rest of the code is unchanged and should work fine now) ...
# 4. Process Canvas Results and Display Metrics
if canvas_result.json_data is not None:
    all_objects = canvas_result.json_data.get("objects", [])
    
    added_points_count = sum(1 for obj in all_objects if obj.get('stroke') == '#00FF00')
    removed_points_count = sum(1 for obj in all_objects if obj.get('stroke') == '#FF0000')
    
    final_count = initial_count + added_points_count - removed_points_count
    
    st.markdown("---")
    st.subheader("Colony Count Summary")
    
    st.metric(label="Initial YOLO Count", value=initial_count)
    st.metric(label="User Added (Green dots)", value=added_points_count)
    st.metric(label="User Marked for Subtraction (Red dots)", value=removed_points_count)
    st.metric(label="**Final Colony Count**", value=final_count, 
              delta=final_count - initial_count, delta_color="normal")
    
    st.markdown("---")

    # 5. Recreate the final annotated OpenCV image with dots and text
    img_final = img_cv_base.copy() 

    # Draw User-Added/Removed Dots (Overlay)
    dot_radius = 5
    for obj in all_objects:
        if obj.get('type') == 'circle':
            center_x = int(obj['left'])
            center_y = int(obj['top'])
            color_hex = obj['stroke']
            
            bgr_color = hex_to_bgr(color_hex.lstrip('#'))
            cv2.circle(img_final, (center_x, center_y), dot_radius, bgr_color, -1)
            
    # Add count text (bottom-right)
    text = f"Colonies: {final_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3 
    thickness = 5
    
    img_h, img_w, _ = img_final.shape

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = img_w - text_size[0] - 10 
    text_y = img_h - 10 
    cv2.putText(img_final, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
    
    # Display the final image
    st.image(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB), caption="Final Annotated Image", use_column_width=True)


    # 6. Download functionality
    is_success, buffer = cv2.imencode(".jpg", img_final)
    if is_success:
        download_data = BytesIO(buffer)

        st.download_button(
            label="Download Final Annotated Image",
            data=download_data,
            file_name=f"{selected_id}_annotated_final.jpg",
            mime="image/jpeg"
        )
