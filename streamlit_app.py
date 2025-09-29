import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from io import BytesIO

# --- Configuration & Initialization ---

# Custom title/icon
try:
    im = Image.open("App_Icon.jpg") 
except FileNotFoundError:
    im = "🧫"
st.set_page_config(page_title="Colony Counter v2 (Editable)", page_icon=im)

# Header
st.title("🧫 Colony Counter v2 (Editable)")

# Load weights (assuming weights.pt is in the same directory)
model = YOLO("weights.pt")
    
# Function to convert YOLO boxes to a simple JSON list of centers for the canvas
def yolo_boxes_to_initial_dots_json(boxes, stroke_color, radius=5):
    """Converts YOLO bounding boxes to 'circle' objects for st_canvas using the center of the box."""
    canvas_objects = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(float, box)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # We store the original box as custom data for redrawing the rectangle later
        canvas_objects.append({
            "type": "circle",
            "left": center_x,
            "top": center_y,
            "radius": radius,
            "fill": stroke_color, # Color of the dot
            "stroke": stroke_color,
            "strokeWidth": 1,
            "originX": "center",
            "originY": "center",
            "point_type": "yolo_detection", # Custom tag for original detections
            "original_box": [x1, y1, x2, y2] # Store box for redrawing
        })
    
    return {"objects": canvas_objects, "background": ""}

# --- Image Upload ---

# upload img
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # Read and convert to OpenCV (BGR)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  
    
    # Convert to PIL (RGB) for st_canvas
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    st.image(img_pil, caption="Original Image", use_column_width=True)

    # --- YOLO Inference and Initial Setup ---

    if st.button("Run YOLO Inference"):
        
        # 1. Run YOLO Inference and prepare initial state
        with st.spinner('Running YOLO inference...'):
            results = model(img_cv, verbose=False)
            yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
            
        initial_drawing = yolo_boxes_to_initial_dots_json(yolo_boxes, "#00FF00") # Green dots
        
        # Store essential data in session state for editing phase
        st.session_state['initial_drawing_json'] = initial_drawing
        st.session_state['img_pil'] = img_pil
        st.session_state['img_cv_original'] = img_cv.copy() # Store a clean original copy
        st.session_state['image_dims'] = img_cv.shape[:2] # (height, width)
        st.session_state['yolo_boxes'] = yolo_boxes # Keep original boxes for redrawing rectangles

# --- Interactive Editing Canvas (Conditional Block) ---

if 'img_pil' in st.session_state:
    st.markdown("---")
    st.subheader("Interactive Colony Editor")
    
    # 2. Setup Drawing Controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        draw_mode = st.radio(
            "Select action:",
            ("Add Colony", "Remove Colony"),
            key="draw_mode",
            help="Click on the image to place a dot. Green dots add to the count, red dots subtract."
        )
        
        if draw_mode == "Add Colony":
            stroke_color = "#00FF00"  # Green
        else: # Remove Colony
            stroke_color = "#FF0000"  # Red
        
    # 3. Display the Canvas
    with col2:
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

    # 4. Process Canvas Results and Display Metrics
    if canvas_result.json_data is not None:
        all_objects = canvas_result.json_data.get("objects", [])
        
        # Recalculate the count
        # Initial points are ALL points from the YOLO run (all green dots marked as 'yolo_detection')
        yolo_points_count = sum(1 for obj in all_objects if obj.get('point_type') == 'yolo_detection')
        
        # User-added points are new green dots (stroke #00FF00, no 'yolo_detection' tag)
        added_points_count = sum(1 for obj in all_objects if obj.get('stroke') == '#00FF00' and obj.get('point_type') != 'yolo_detection')
        
        # User-removed points are red dots (stroke #FF0000)
        removed_points_count = sum(1 for obj in all_objects if obj.get('stroke') == '#FF0000')
        
        final_count = yolo_points_count + added_points_count - removed_points_count
        
        st.markdown("---")
        st.subheader("Colony Count Summary")
        
        colony_count_initial = len(st.session_state.get('yolo_boxes', []))
        st.metric(label="Initial YOLO Count", value=colony_count_initial)
        st.metric(label="User Added (Green dots)", value=added_points_count)
        st.metric(label="User Marked for Removal (Red dots)", value=removed_points_count)
        st.metric(label="**Final Colony Count**", value=final_count, 
                  delta=final_count - colony_count_initial, delta_color="normal")
        
        st.markdown("---")

        # 5. Recreate the final annotated OpenCV image with both boxes and text
        
        # Start with a clean copy of the original CV image
        img_annotated = st.session_state['img_cv_original'].copy()
        
        # --- Draw YOLO Bounding Boxes ---
        # The bounding boxes are drawn using the original box data stored in session state
        yolo_boxes = st.session_state.get('yolo_boxes', [])
        for box in yolo_boxes:
            x1, y1, x2, y2 = map(int, box)
            # Draw green bounding boxes
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # --- Draw User-Added/Removed Dots (Overlay) ---
        # The canvas only provides the final image with dots *on top of a blank image*.
        # To get the dots *on top of the boxes*, we must use OpenCV's drawing functions 
        # based on the coordinates of the dots from the JSON data.
        
        # This is more complex, so a simpler/faster method is to use the canvas image data itself,
        # but since that lacks the boxes, we'll draw the dots onto the OpenCV image with the boxes.
        
        dot_radius = 5 # Matches the point_display_radius in the canvas
        for obj in all_objects:
            if obj.get('type') == 'circle':
                center_x = int(obj['left'])
                center_y = int(obj['top'])
                color_hex = obj['stroke']
                
                # Convert hex color to BGR for OpenCV
                hex_to_bgr = lambda h: tuple(int(h[i:i+2], 16) for i in (4, 2, 0)) # BGR order
                
                # Only draw new (non-YOLO) points, as the YOLO points are already represented by the boxes
                # OR draw ALL points (YOLO included) as dots for clarity. Let's draw all user-placed dots.
                
                # Only draw the RED (removed) and new GREEN (added) dots to show user modifications.
                if obj.get('point_type') != 'yolo_detection':
                    bgr_color = hex_to_bgr(color_hex.lstrip('#'))
                    # Draw a solid circle (thickness=-1)
                    cv2.circle(img_annotated, (center_x, center_y), dot_radius, bgr_color, -1)
                
        # --- Add count text (bottom-right) ---
        text = f"Colonies: {final_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3 
        thickness = 5
        
        img_h, img_w, _ = img_annotated.shape

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = img_w - text_size[0] - 10 
        text_y = img_h - 10 
        cv2.putText(img_annotated, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness) # Green text (BGR: 0, 255, 0)
        
        # Display the final image
        st.image(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB), caption="Final Annotated Image (Boxes and Dots)", use_column_width=True)


        # 6. Save and Download functionality (re-using original logic)
        save_path = "annotated_streamlit.jpg"
        cv2.imwrite(save_path, img_annotated)
        st.success(f"Annotated image saved as {save_path}")
        
        with open(save_path, "rb") as file:
            st.download_button(
                label="Download Final Annotated Image",
                data=file.read(),
                file_name="annotated_image.jpg",
                mime="image/jpeg"
            )
