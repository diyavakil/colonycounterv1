import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates # New Import

# --- Initialization and Configuration ---

# custom title/icon
im = Image.open("App_Icon.jpg") # cute squid
st.set_page_config(page_title="Colony Counter v1", page_icon=im, layout="wide")

# header
st.title("🧫 Colony Counter v1")

model = YOLO("weights.pt") # load weights

# --- Session State Management ---

# Initialize session state variables if they don't exist
if 'yolo_count' not in st.session_state:
    st.session_state['yolo_count'] = 0
if 'manual_points' not in st.session_state:
    st.session_state['manual_points'] = []
if 'img_annotated' not in st.session_state:
    st.session_state['img_annotated'] = None
if 'img_original_cv' not in st.session_state:
    st.session_state['img_original_cv'] = None
if 'inference_run' not in st.session_state:
    st.session_state['inference_run'] = False


# Function to handle YOLO inference
def run_yolo_inference(img):
    """Runs YOLO inference and initializes the session state with results."""
    
    # Reset manual additions
    st.session_state['manual_points'] = []
    
    results = model(img)
    img_annotated = img.copy()

    # Draw YOLO bboxes (green rectangles)
    yolo_bboxes = results[0].boxes.xyxy.cpu().numpy()
    for box in yolo_bboxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Store results in session state
    st.session_state['yolo_count'] = len(yolo_bboxes)
    st.session_state['img_annotated'] = img_annotated
    st.session_state['inference_run'] = True

# Function to draw the final image with count and manual points
def draw_final_image():
    """Draws the final annotated image including manual points and count."""
    
    if st.session_state['img_annotated'] is None:
        return None

    # Start with the image that has YOLO bboxes
    final_img = st.session_state['img_annotated'].copy()
    
    # Draw manual points (red dots)
    for x, y in st.session_state['manual_points']:
        # Draw a small red circle
        cv2.circle(final_img, (x, y), 5, (0, 0, 255), -1) 
    
    # Calculate total count
    total_count = st.session_state['yolo_count'] + len(st.session_state['manual_points'])
    
    # Add total count in bottom-right corner (Green for YOLO + Manual)
    text = f"Total Colonies: {total_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5 # Adjusted for better visibility with the component
    thickness = 3 # Adjusted
    
    # Get text size for background box and positioning
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = final_img.shape[1] - text_size[0] - 10 
    text_y = final_img.shape[0] - 10 
    
    # Optional: Draw a background rectangle for better text visibility
    cv2.rectangle(final_img, (text_x - 5, text_y - text_size[1] - 5), 
                  (final_img.shape[1], final_img.shape[0]), (0, 0, 0), -1) # Black background
    
    # Draw text
    cv2.putText(final_img, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

    return final_img

# --- Streamlit Layout ---

# upload img
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Handle file upload
if uploaded_file is not None:
    # Read the file bytes and convert to OpenCV image once per upload
    if st.session_state['img_original_cv'] is None or uploaded_file.name != st.session_state.get('uploaded_file_name'):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.session_state['img_original_cv'] = img
        st.session_state['uploaded_file_name'] = uploaded_file.name # Store name for check
        
        # Reset everything for a new image
        st.session_state['yolo_count'] = 0
        st.session_state['manual_points'] = []
        st.session_state['img_annotated'] = None
        st.session_state['inference_run'] = False
        
    img_original = st.session_state['img_original_cv']
    
    st.image(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
    
    # Button to run inference
    if st.button("Run YOLO Inference", key='run_yolo'):
        run_yolo_inference(img_original)
        # Rerun to display initial annotated image
        st.experimental_rerun()
        
    # Only show interactive component after inference is run
    if st.session_state['inference_run']:
        
        st.subheader("Manual Correction 🖱️")
        st.info("Click the image below to manually add a colony (a red dot will appear). The total count updates automatically.")

        # Create the image to be displayed in the clickable component
        image_for_click = draw_final_image()
        image_for_click_rgb = cv2.cvtColor(image_for_click, cv2.COLOR_BGR2RGB)
        
        # Display the image using the component to get coordinates
        # The key is crucial: changing it forces the component to remount and re-render the image
        clicked_value = streamlit_image_coordinates(
            image_for_click_rgb, 
            key=f"image_coordinates_{len(st.session_state['manual_points'])}", # Key updates on manual point addition
            use_column_width=True
        )
        
        # Check if a new click happened
        if clicked_value is not None:
            
            # Extract x, y from the dictionary returned by the component
            x_coord = clicked_value['x']
            y_coord = clicked_value['y']
            
            # Add the new point to the list of manual points
            new_point = (x_coord, y_coord)
            
            # Simple check to avoid adding the exact same point multiple times on a rerun
            if new_point not in st.session_state['manual_points']:
                st.session_state['manual_points'].append(new_point)
                
                # Force a rerun to update the image with the new point and count
                st.experimental_rerun()
                
        # --- Save and Download ---
        
        if st.session_state['img_annotated'] is not None:
            
            final_image_to_save = draw_final_image()

            # Save img to file (Need to save the final image with manual points)
            save_path = "annotated_streamlit.jpg"
            cv2.imwrite(save_path, final_image_to_save)
            
            st.markdown("---")
            st.success(f"Final total colonies: **{st.session_state['yolo_count'] + len(st.session_state['manual_points'])}**")
            
            with open(save_path, "rb") as file:
                st.download_button(
                    label="Download Final Annotated Image",
                    data=file.read(),
                    file_name="annotated_image_with_manual_corrections.jpg",
                    mime="image/jpeg"
                )
