import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates 

# --- Streamlit Session State Initialization ---
# Initialize necessary state variables right at the start
if 'inference_run' not in st.session_state:
    st.session_state['inference_run'] = False
if 'img_annotated' not in st.session_state:
    st.session_state['img_annotated'] = None
if 'manual_points' not in st.session_state:
    st.session_state['manual_points'] = []
if 'yolo_count' not in st.session_state:
    st.session_state['yolo_count'] = 0
if 'img_original_cv' not in st.session_state:
    st.session_state['img_original_cv'] = None
if 'uploaded_file_id' not in st.session_state:
    st.session_state['uploaded_file_id'] = None

# custom title/icon
im = Image.open("App_Icon.jpg") # cute squid
st.set_page_config(page_title="Colony Counter v1", page_icon=im)

# header
st.title("🧫 Colony Counter v1")

model = YOLO("weights.pt") # load weights


# --- Callback Function for Inference Button ---
def run_inference_callback(img):
    """Callback function to run YOLO and store initial results in session state."""
    
    # Run YOLO and get results
    results = model(img)
    img_annotated = img.copy()
    
    # Draw YOLO bboxes
    yolo_bboxes = results[0].boxes.xyxy.cpu().numpy()
    for box in yolo_bboxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Store results in session state
    st.session_state['yolo_count'] = len(yolo_bboxes)
    st.session_state['img_annotated'] = img_annotated
    st.session_state['manual_points'] = [] # Reset manual points on new inference
    st.session_state['inference_run'] = True

# --- Function to Draw the Final Image (In-Memory) ---
@st.cache_data(show_spinner=False)
def draw_final_image_with_manual_points(img_base, manual_points, yolo_count):
    """Draws the final annotated image including manual points and count."""
    
    # Start with the base image that has YOLO bboxes
    final_img = img_base.copy()
    
    # Draw manual points (red dots)
    for x, y in manual_points:
        # Draw a small red circle (colony dot)
        cv2.circle(final_img, (x, y), 5, (0, 0, 255), -1) 
    
    # Calculate total count
    total_count = yolo_count + len(manual_points)
    
    # Add total count in bottom-right corner
    text = f"Total Colonies: {total_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5 
    thickness = 3 
    
    # Get text size and position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = final_img.shape[1] - text_size[0] - 10 
    text_y = final_img.shape[0] - 10 
    
    # Draw a background rectangle for better text visibility
    cv2.rectangle(final_img, (text_x - 5, text_y - text_size[1] - 5), 
                  (final_img.shape[1], final_img.shape[0]), (0, 0, 0), -1) 
    
    # Draw text (Green)
    cv2.putText(final_img, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

    # Convert to RGB for Streamlit display and return the count
    return cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), final_img, total_count


# --- Main Application Logic ---

# upload img
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Check if a new file is uploaded
    if st.session_state['uploaded_file_id'] is None or st.session_state['uploaded_file_id'] != uploaded_file.file_id:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.session_state['img_original_cv'] = img
        st.session_state['uploaded_file_id'] = uploaded_file.file_id
        st.session_state['inference_run'] = False # Reset inference on new image
        
    img_original = st.session_state['img_original_cv']

    st.image(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
    
    # Button calls the callback function, which updates session state
    st.button(
        "Run YOLO Inference", 
        key='run_yolo', 
        on_click=run_inference_callback, 
        args=(img_original,)
    )

    # --- Interactive Manual Correction Section ---
    if st.session_state['inference_run']:
        
        st.markdown("---")
        st.subheader("Manual Colony Correction 🖱️")
        st.info(f"YOLO detected **{st.session_state['yolo_count']}** colonies. Click on the image below to manually add missing colonies.")

        # 1. Get the current state of the annotated image
        # This returns the RGB image for the web, the BGR image for saving, and the final count.
        final_img_rgb, final_img_bgr, final_count = draw_final_image_with_manual_points(
            st.session_state['img_annotated'],
            st.session_state['manual_points'],
            st.session_state['yolo_count']
        )
        
        # 2. Display the image with the custom component and capture clicks
        clicked_value = streamlit_image_coordinates(
            final_img_rgb, 
            key="image_clicker", 
            use_column_width=True
        )

        # 3. Process the click
        if clicked_value is not None:
            
            x_coord = clicked_value['x']
            y_coord = clicked_value['y']
            new_point = (x_coord, y_coord)
            
            # Add to the list in session state
            if new_point not in st.session_state['manual_points']:
                st.session_state['manual_points'].append(new_point)
                
                # Force rerun to redraw the image with the new point and count
                st.experimental_rerun()
                
        # --- Final Count and Download ---
        st.markdown("---")
        st.success(f"Final Total Colonies: **{final_count}**")

        # Save and Download option, only once after the clicking is done
        save_path = "annotated_streamlit_final.jpg"
        
        # Use a button to trigger the save/download *after* the user is finished clicking
        if st.button("Save and Download Annotated Image", key="download_btn"):
            cv2.imwrite(save_path, final_img_bgr) 
            
            with open(save_path, "rb") as file:
                st.download_button(
                    label="Download Annotated Image",
                    data=file.read(),
                    file_name="annotated_image_final.jpg",
                    mime="image/jpeg"
                )
