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
# and initial drawing on every click. We store the result in a PIL format for the canvas.
@st.cache_data
def run_initial_inference(img, model):
    """Runs YOLO inference and returns the initial annotated image (CV2) and count."""
    results = model(img)
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
    manual_count = 0
    if manual_points_data and 'objects' in manual_points_data:
        for obj in manual_points_data['objects']:
            if obj.get('type') == 'circle':
                # The canvas provides coordinates relative to the object center
                # We approximate the center of the circle drawn by the user
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
    text_x = final_img.shape[1] - text_size[0] - 10 
    text_y = final_img.shape[0] - 10 
    
    # Draw black background rectangle
    cv2.rectangle(final_img, (text_x - 5, text_y - text_size[1] - 5), 
                  (final_img.shape[1], final_img.shape[0]), (0, 0, 0), -1) 
    
    # Draw text (Green)
    cv2.putText(final_img, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

    return final_img, total_count


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
        run_initial_inference.clear() # Clear cache for new file
        st.session_state['inference_run'] = False
        
    img_original = st.session_state['img_original_cv']

    st.image(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
    
    # Button calls the callback function
    st.button(
        "Run YOLO Inference", 
        key='run_yolo', 
        on_click=run_inference_callback, 
        args=(img_original, model)
    )

    # ---------------------------------------------
    # --- Interactive Manual Correction Section ---
    # ---------------------------------------------
    if st.session_state['inference_run']:
        
        st.markdown("---")
        st.subheader("Manual Colony Correction 🖱️")
        st.info(f"YOLO detected **{st.session_state['yolo_count']}** colonies. Draw a small **red circle** on the image below to manually add missing colonies.")
        
        # 1. Use st_canvas to display the image and allow drawing
        # The height and width must be set explicitly using the image dimensions
        canvas_result = st_canvas(
            # Pass the PIL Image with YOLO annotations as background
            background_image=st.session_state['img_annotated_pil'],
            
            # Set the tool to draw circles (for colonies)
            drawing_mode="circle",
            
            # Set drawing color to Red (for contrast with YOLO's green)
            stroke_color="rgba(255, 0, 0, 1)", # Red with 100% opacity
            
            # Set stroke width small, like a dot
            stroke_width=5, 
            
            # Use the actual image dimensions for the canvas size
            height=st.session_state['img_annotated_pil'].height,
            width=st.session_state['img_annotated_pil'].width,
            
            # Use columns to allow the canvas to stretch
            use_container_width=True, 
            
            # Critical: The component returns the drawing data here (JSON)
            key="colony_canvas" 
        )

        # 2. Process the results for live count display
        manual_points_data = canvas_result.json_data
        manual_count = 0
        
        if manual_points_data and 'objects' in manual_points_data:
            # Count the number of objects (circles) drawn by the user
            manual_count = len(manual_points_data['objects'])
            
            # Store the data for final download
            st.session_state['canvas_json_data'] = manual_points_data

        final_count = st.session_state['yolo_count'] + manual_count
        
        # --- Live Final Count Display ---
        st.markdown("---")
        st.success(f"**Live Total Colonies: {final_count}** (YOLO: {st.session_state['yolo_count']} + Manual: {manual_count})")

        # 3. Download Logic
        # This button triggers the final drawing of the count text and manual points onto the CV2 image
        if st.button("Save and Download Final Annotated Image", key="download_btn"):
            
            # Get the final image with all annotations (YOLO boxes, manual dots, and final count text)
            final_image_to_save, final_count = finalize_image_for_download(
                st.session_state['img_annotated_cv2'],
                st.session_state['canvas_json_data'],
                st.session_state['yolo_count']
            )

            # Save the final image to a file
            save_path = "annotated_streamlit_final.jpg"
            cv2.imwrite(save_path, final_image_to_save) 
            
            st.success(f"Annotated image saved as {save_path}")
            
            with open(save_path, "rb") as file:
                st.download_button(
                    label="Download Annotated Image",
                    data=file.read(),
                    file_name="annotated_image_final.jpg",
                    mime="image/jpeg"
                )
