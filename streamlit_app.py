import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# custom title/icon
im = Image.open("App_Icon.jpg") # cute squid
st.set_page_config(page_title="Colony Counter v1", page_icon=im)

# header
st.title("🧫 Colony Counter v1")

model = YOLO("weights.pt") # load weights

# upload img
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # convert uploaded file to OpenCV img
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) 
    
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
    
    if st.button("Run YOLO Inference"):
        results = model(img)
        img_annotated = img.copy()
        
        # FIX START: Converted tensor to iterable numpy array and fixed indentation
        yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # draw only the little green bboxes
        for box in yolo_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        
        # count colonies
        colony_count = len(yolo_boxes)
        
        # add count in bottom-right corner
        text = f"Colonies: {colony_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3 # text size
        thickness = 5 # text thickness
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = img_annotated.shape[1] - text_size[0] - 10  # 10 px from right
        text_y = img_annotated.shape[0] - 10  # 10 px from bottom
        cv2.putText(img_annotated, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
        
        st.image(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_column_width=True)

        # save img to file
        save_path = "annotated_streamlit.jpg"
        cv2.imwrite(save_path, img_annotated)
        st.success(f"Annotated image saved as {save_path}")
        st.download_button(
            label="Download Annotated Image",
            data=open(save_path, "rb").read(),
            file_name="annotated_image.jpg",
            mime="image/jpeg"
        )








# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from PIL import Image
# from streamlit_drawable_canvas import st_canvas
# from io import BytesIO
# import base64

# # --- Helper Function for Base64 Encoding ---
# def pil_to_base64_url(img_pil):
#     """Converts a PIL Image object to a Base64 URL string for st_canvas."""
#     buffered = BytesIO()
#     img_pil.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     return f"data:image/png;base64,{img_str}"

# # --- Configuration & Initialization ---
# try:
#     im = Image.open("App_Icon.jpg") 
# except FileNotFoundError:
#     im = "🧫"
# st.set_page_config(page_title="Colony Counter v2 (Editable)", page_icon=im)

# st.title("🧫 Colony Counter v2 (Editable)")

# model = YOLO("weights.pt")

# # Function to convert YOLO boxes to initial dots
# def yolo_boxes_to_initial_dots_json(boxes, stroke_color, radius=5):
#     canvas_objects = []
#     for box in boxes:
#         x1, y1, x2, y2 = map(float, box)
#         center_x = (x1 + x2) / 2
#         center_y = (y1 + y2) / 2
        
#         canvas_objects.append({
#             "type": "circle",
#             "left": center_x,
#             "top": center_y,
#             "radius": radius,
#             "fill": stroke_color,
#             "stroke": stroke_color,
#             "strokeWidth": 1,
#             "originX": "center",
#             "originY": "center",
#             "point_type": "yolo_detection",
#             "original_box": [x1, y1, x2, y2]
#         })
#     return {"objects": canvas_objects, "background": ""}

# # --- Image Upload ---
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
    
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  
    
#     # Convert to PIL (RGB) for display and Base64 conversion
#     img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
#     st.image(img_pil, caption="Original Image", use_column_width=True)

#     # --- YOLO Inference and Initial Setup ---
#     if st.button("Run YOLO Inference"):
        
#         with st.spinner('Running YOLO inference...'):
#             results = model(img_cv, verbose=False)
#             yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
            
#         initial_drawing = yolo_boxes_to_initial_dots_json(yolo_boxes, "#00FF00")
        
#         # --- KEY FIX: Encode PIL image to Base64 URL ---
#         base64_url = pil_to_base64_url(img_pil) 
        
#         st.session_state['initial_drawing_json'] = initial_drawing
#         st.session_state['base64_background'] = base64_url # Store Base64 string
#         st.session_state['img_cv_original'] = img_cv.copy()
#         st.session_state['image_dims'] = img_cv.shape[:2]
#         st.session_state['yolo_boxes'] = yolo_boxes
#         st.session_state['yolo_inferred'] = True
#         st.experimental_rerun()

# # --- Interactive Editing Canvas (Conditional Block) ---

# if st.session_state.get('yolo_inferred', False) and 'image_dims' in st.session_state:
#     st.markdown("---")
#     st.subheader("Interactive Colony Editor")
    
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         if 'draw_mode' not in st.session_state:
#              st.session_state['draw_mode'] = "Add Colony"

#         draw_mode = st.radio(
#             "Select action:",
#             ("Add Colony", "Remove Colony"),
#             key="draw_mode",
#             help="Click on the image to place a dot. Green dots add to the count, red dots subtract."
#         )
        
#         stroke_color = "#00FF00" if draw_mode == "Add Colony" else "#FF0000"
        
#     # 3. Display the Canvas
#     with col2:
#         canvas_result = st_canvas(
#             fill_color="rgba(0, 0, 0, 0.0)",
#             stroke_width=5, 
#             stroke_color=stroke_color,
#             # --- KEY FIX: Use Base64 URL for background image ---
#             # NOTE: background_image here expects a PIL image OR a Base64 URL string
#             background_image=st.session_state['base64_background'], 
#             # ----------------------------------------------------
#             initial_drawing=st.session_state['initial_drawing_json'],
#             update_streamlit=True,
#             height=st.session_state['image_dims'][0], 
#             width=st.session_state['image_dims'][1],
#             drawing_mode="point",
#             point_display_radius=5,
#             display_toolbar=False,
#             key="colony_editor_canvas", 
#         )

#     # --- Processing and Metrics (Unchanged) ---
#     if canvas_result.json_data is not None:
#         all_objects = canvas_result.json_data.get("objects", [])
        
#         yolo_points_count = sum(1 for obj in all_objects if obj.get('point_type') == 'yolo_detection')
#         added_points_count = sum(1 for obj in all_objects if obj.get('stroke') == '#00FF00' and obj.get('point_type') != 'yolo_detection')
#         removed_points_count = sum(1 for obj in all_objects if obj.get('stroke') == '#FF0000')
        
#         final_count = yolo_points_count + added_points_count - removed_points_count
        
#         st.markdown("---")
#         st.subheader("Colony Count Summary")
        
#         colony_count_initial = len(st.session_state.get('yolo_boxes', []))
#         st.metric(label="Initial YOLO Count", value=colony_count_initial)
#         st.metric(label="User Added (Green dots)", value=added_points_count)
#         st.metric(label="User Marked for Removal (Red dots)", value=removed_points_count)
#         st.metric(label="**Final Colony Count**", value=final_count, 
#                   delta=final_count - colony_count_initial, delta_color="normal")
        
#         st.markdown("---")

#         # 5. Recreate the final annotated OpenCV image with both boxes and text
#         img_annotated = st.session_state['img_cv_original'].copy()
        
#         # --- Draw YOLO Bounding Boxes ---
#         yolo_boxes = st.session_state.get('yolo_boxes', [])
#         for box in yolo_boxes:
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
#         # --- Draw User-Added/Removed Dots (Overlay) ---
#         dot_radius = 5
#         for obj in all_objects:
#             if obj.get('type') == 'circle':
#                 center_x = int(obj['left'])
#                 center_y = int(obj['top'])
#                 color_hex = obj['stroke']
                
#                 hex_to_bgr = lambda h: tuple(int(h[i:i+2], 16) for i in (4, 2, 0))
                
#                 # Only draw new green (added) and red (removed) dots
#                 if obj.get('point_type') != 'yolo_detection':
#                     bgr_color = hex_to_bgr(color_hex.lstrip('#'))
#                     cv2.circle(img_annotated, (center_x, center_y), dot_radius, bgr_color, -1)
                
#         # --- Add count text (bottom-right) ---
#         text = f"Colonies: {final_count}"
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 3 
#         thickness = 5
        
#         img_h, img_w, _ = img_annotated.shape

#         text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
#         text_x = img_w - text_size[0] - 10 
#         text_y = img_h - 10 
#         cv2.putText(img_annotated, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
        
#         # Display the final image
#         st.image(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB), caption="Final Annotated Image (Boxes and Dots)", use_column_width=True)


#         # 6. Save and Download functionality
#         save_path = "annotated_streamlit.jpg"
#         cv2.imwrite(save_path, img_annotated)
#         st.success(f"Annotated image saved as {save_path}")
        
#         with open(save_path, "rb") as file:
#             st.download_button(
#                 label="Download Final Annotated Image",
#                 data=file.read(),
#                 file_name="annotated_image.jpg",
#                 mime="image/jpeg"
#             )
