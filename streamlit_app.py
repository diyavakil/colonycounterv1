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
# Load App Icon (assuming App_Icon.jpg is in the same directory)
try:
    im = Image.open("App_Icon.jpg")
except FileNotFoundError:
    im = None # Use default icon if not found
st.set_page_config(page_title="Colony Counter v2 (Editable)", page_icon=im if im else "🧫")

# Header
st.title("🧫 Colony Counter v2 (Editable)")

# Load YOLO Model (assuming weights.pt is in the same directory)
try:
    model = YOLO("weights.pt")
except Exception as e:
    st.error(f"Error loading YOLO model: {e}. Please ensure 'weights.pt' is in the directory.")
    model = None # Set model to None if loading fails

# Function to convert YOLO boxes to canvas points
def yolo_boxes_to_canvas_points(boxes, stroke_color):
    """Converts YOLO bounding boxes to 'point' objects for st_canvas, using the center of the box."""
    canvas_objects = []
    # Use 'radius' as the half-width/height of the bounding box for visualization
    # We'll approximate with a fixed radius for a 'dot' appearance, as exact box dimensions are complex for a simple dot.
    point_radius = 5 
    
    for box in boxes:
        x1, y1, x2, y2 = map(float, box)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        canvas_objects.append({
            "type": "circle",
            "version": "5.3.0",
            "originX": "center",
            "originY": "center",
            "left": center_x,
            "top": center_y,
            "width": point_radius * 2,
            "height": point_radius * 2,
            "fill": stroke_color, # Initial points are green
            "stroke": stroke_color,
            "strokeWidth": 1,
            "rx": point_radius,
            "ry": point_radius,
            "radius": point_radius,
            "scaleX": 1,
            "scaleY": 1,
            "angle": 0,
            "point_type": "yolo_detection" # Custom field to identify original detections
        })
    
    # Structure needed for initial_drawing
    return {"objects": canvas_objects, "background": ""}

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV img
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  
    
    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
    
    # --- YOLO Inference and Initial Setup ---
    if model and st.button("Run YOLO Inference"):
        # Run inference
        results = model(img_cv, verbose=False)
        yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
        initial_colony_count = len(yolo_boxes)

        # Convert the original image to PIL Image for st_canvas background
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Prepare initial canvas drawing data from YOLO results
        initial_drawing = yolo_boxes_to_canvas_points(yolo_boxes, "#00FF00") # Green for initial detections
        
        # Store state for re-runs
        st.session_state['initial_colony_count'] = initial_colony_count
        st.session_state['initial_drawing_json'] = initial_drawing
        st.session_state['img_pil'] = img_pil
        
    
    # --- Interactive Editing Canvas ---
    if 'img_pil' in st.session_state:
        st.markdown("---")
        st.subheader("Interactive Colony Editor")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Drawing Mode Selector
            draw_mode = st.radio(
                "Select action:",
                ("Add Colony", "Remove Colony"),
                key="draw_mode"
            )
            
            # Configure colors based on mode
            if draw_mode == "Add Colony":
                stroke_color = "#00FF00"  # Green for adding
            else: # Remove Colony
                stroke_color = "#FF0000"  # Red for removing

            st.write(f"Click on the image to {draw_mode.lower()} (Dot color: {stroke_color})")
            
        # Display the canvas
        with col2:
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0.0)", # No fill
                stroke_width=5, 
                stroke_color=stroke_color,
                background_image=st.session_state['img_pil'],
                initial_drawing=st.session_state['initial_drawing_json'],
                update_streamlit=True, # Realtime update
                height=st.session_state['img_pil'].height,
                width=st.session_state['img_pil'].width,
                drawing_mode="point", # Only point drawing is allowed
                point_display_radius=5, # Small green/red dot
                display_toolbar=False, # No need for full toolbar
                key="colony_editor_canvas",
            )

        st.markdown("---")

        # --- Count and Display Results ---
        if canvas_result.json_data is not None:
            all_objects = canvas_result.json_data.get("objects", [])
            
            # Recalculate the count
            # 1. Count initial YOLO detections (green points from initial_drawing)
            yolo_points_count = sum(1 for obj in all_objects if obj.get('point_type') == 'yolo_detection')
            
            # 2. Count user-added points (new green points)
            added_points_count = sum(1 for obj in all_objects if obj.get('stroke') == '#00FF00' and obj.get('point_type') != 'yolo_detection')
            
            # 3. Count user-removed points (new red points)
            removed_points_count = sum(1 for obj in all_objects if obj.get('stroke') == '#FF0000')
            
            # The final count is the YOLO detections plus user additions, minus user removals
            final_count = yolo_points_count + added_points_count - removed_points_count
            
            st.metric(label="Initial YOLO Colony Count", value=st.session_state.get('initial_colony_count', 0))
            st.metric(label="User Added Colonies (Green dots)", value=added_points_count)
            st.metric(label="User Removed Colonies (Red dots)", value=removed_points_count)
            st.metric(label="**Final Colony Count**", value=final_count, delta=final_count - st.session_state.get('initial_colony_count', 0), delta_color="normal")


            # --- Download Button for the final annotated image ---
            if canvas_result.image_data is not None:
                # Get the final image data from the canvas result
                img_final_np = canvas_result.image_data
                
                # Convert RGBA numpy array to JPEG for download
                img_final_pil = Image.fromarray(img_final_np.astype('uint8'), 'RGBA').convert('RGB')
                
                # Add the count text to the bottom right of the image (re-using PIL for simplicity here)
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img_final_pil)
                text = f"Colonies: {final_count}"
                
                # Try to use a common font, or default if not found
                try:
                    font = ImageFont.truetype("arial.ttf", size=50) # Common font
                except IOError:
                    font = ImageFont.load_default()
                
                # Calculate text size and position
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                margin = 20
                text_x = img_final_pil.width - text_width - margin
                text_y = img_final_pil.height - text_height - margin
                
                # Draw a simple background for contrast if needed, or just text
                draw.text((text_x, text_y), text, fill=(0, 255, 0), font=font) # Green text

                # Save to a buffer to create a download button
                buf = BytesIO()
                img_final_pil.save(buf, format="JPEG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="Download Final Annotated Image",
                    data=byte_im,
                    file_name="final_annotated_colonies.jpg",
                    mime="image/jpeg"
                )
