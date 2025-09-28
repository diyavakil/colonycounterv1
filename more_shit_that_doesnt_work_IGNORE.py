# from streamlit_drawable_canvas import st_canvas
# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from PIL import Image
# import os # Used for checking if weights.pt exists

# # --- 1. CONFIGURATION AND INITIALIZATION ---

# st.set_page_config(
#     page_title="Colony Counter v1",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
# st.title("🧫 Automated Colony Counter v1")

# # --- 2. CACHED MODEL LOADING ---

# @st.cache_resource
# def load_yolo_model(path: str):
#     """Loads the YOLO model only once for efficiency."""
#     if not os.path.exists(path):
#         st.error(f"Model file not found at: '{path}'")
#         st.error("Please ensure 'weights.pt' is in the root directory of your Streamlit app.")
#         st.stop()
#     try:
#         # We assume the model is trained to detect 'colony'
#         model = YOLO(path)
#         return model
#     except Exception as e:
#         st.error(f"Error initializing YOLO model from '{path}': {e}")
#         st.stop()

# # Load the model
# model = load_yolo_model("weights.pt")

# # Initialize session state variables if they don't exist
# if "img_annotated" not in st.session_state:
#     st.session_state["img_annotated"] = None
# if "colony_count" not in st.session_state:
#     st.session_state["colony_count"] = 0
# if "last_correction_mode" not in st.session_state:
#     st.session_state["last_correction_mode"] = "None"
# if "drawing_json" not in st.session_state: # CRITICAL: Store the full JSON drawing data for persistence
#     st.session_state["drawing_json"] = None
# if "last_correction_objects" not in st.session_state:
#     st.session_state["last_correction_objects"] = [] 


# # --- 3. SIDEBAR CONTROLS ---

# st.sidebar.header("User Correction Tools")
# st.sidebar.markdown("Use this section to manually adjust the count after inference.")
# mode = st.sidebar.radio(
#     "Correction Mode",
#     ["None", "➕ Add Colony", "➖ Remove Colony"],
#     index=0,
#     help="Select 'Add' to mark missed colonies (green dot), or 'Remove' to mark false positives (red dot)."
# )
# # Update the mode in session state to persist it across reruns
# st.session_state["last_correction_mode"] = mode


# # --- 4. FILE UPLOAD AND YOLO INFERENCE ---

# st.subheader("1. Upload Image & Run Detection")
# uploaded_file = st.file_uploader("Upload a petri dish image", type=["jpg", "jpeg", "png"])

# img = None
# if uploaded_file is not None:
#     # Read file and decode into OpenCV format (BGR)
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     if img is not None:
#         st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

#         # Disable button after initial run to prevent state inconsistencies, but allow rerun when new image is uploaded
#         if st.button("Run YOLO Inference", type="primary", disabled=st.session_state["img_annotated"] is not None):
#             with st.spinner('Running detection on the image... this may take a moment.'):
#                 # 4.1. Run detection
#                 results = model(img)
            
#             img_annotated = img.copy()
#             colony_count = 0

#             # 4.2. Process and draw results
#             if results and len(results) > 0 and results[0].boxes:
#                 # Use CPU for drawing as it is faster outside the YOLO process
#                 boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                
#                 for box in boxes:
#                     x1, y1, x2, y2 = box
#                     # Draw a green bounding box (BGR format)
#                     cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
#                 colony_count = len(boxes)
#                 st.success(f"YOLO detection complete! Initial count: **{colony_count}**")
#             else:
#                 st.info("YOLO did not detect any colonies in this image.")
            
#             # 4.3. Save results and reset drawing state
#             st.session_state["img_annotated"] = img_annotated
#             st.session_state["colony_count"] = colony_count
#             st.session_state["drawing_json"] = None # IMPORTANT: Clear drawing on new inference
#             st.session_state["last_correction_objects"] = []
#     else:
#         st.error("Could not decode the image file. Please try a different format.")


# # --- 5. CORRECTION CANVAS AND FINAL COUNT (REWRITTEN FOR STABILITY) ---

# if st.session_state["img_annotated"] is not None:
    
#     st.subheader("2. Review and Correct Predictions")

#     img_annotated = st.session_state["img_annotated"]
    
#     # 5.1. Prepare image for canvas
#     if img_annotated.shape[0] > 0 and img_annotated.shape[1] > 0:
#         background_img_pil = Image.fromarray(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
        
#         # Determine canvas drawing parameters based on sidebar mode
#         current_drawing_mode = "point" if st.session_state["last_correction_mode"] != "None" else "none"
        
#         # Color configuration (determines the color of the point drawn on the next click)
#         if st.session_state["last_correction_mode"] == "➕ Add Colony":
#             stroke_color = "green"
#             fill_color = "rgba(0, 255, 0, 0.9)"
#         elif st.session_state["last_correction_mode"] == "➖ Remove Colony":
#             stroke_color = "red"
#             fill_color = "rgba(255, 0, 0, 0.9)"
#         else:
#             stroke_color = "gray"
#             fill_color = "rgba(100, 100, 100, 0.5)"


#         # 5.2. Display Canvas
#         st.markdown(f"**Current Correction Action:** {st.session_state['last_correction_mode']}")
        
#         # Pass the last saved JSON back into initial_drawing for persistence
#         canvas_result = st_canvas(
#             fill_color=fill_color,
#             stroke_width=10,
#             stroke_color=stroke_color,
#             background_image=background_img_pil,
#             initial_drawing=st.session_state["drawing_json"], # CRITICAL FIX: Ensures drawings persist and stabilizes the component
#             update_streamlit=True,
#             height=img_annotated.shape[0] * 0.5, # Keep scaling for better display
#             width=img_annotated.shape[1] * 0.5,  
#             drawing_mode=current_drawing_mode,
#             key="correction_canvas"
#         )
        
#         # 5.3. Calculate Corrections and Final Count
        
#         added_count = 0
#         removed_count = 0
        
#         # Only update the stored JSON if the canvas returned a valid result
#         if canvas_result.json_data is not None:
#              # This is the new drawing state, save it immediately
#             st.session_state["drawing_json"] = canvas_result.json_data

#         # Use the stored JSON to calculate the counts
#         if st.session_state["drawing_json"] and "objects" in st.session_state["drawing_json"]:
            
#             # Iterate through all objects ever drawn
#             for obj in st.session_state["drawing_json"]["objects"]:
#                 # Check for "point" type drawings
#                 if obj.get("type") == "point":
#                     # We check the stroke color saved in the object data to determine if it's an Add or Remove
#                     if obj.get("stroke") == "green":
#                         added_count += 1
#                     elif obj.get("stroke") == "red":
#                         removed_count += 1
        
        
#         initial_count = st.session_state["colony_count"]
#         corrected_count = max(0, initial_count + added_count - removed_count)


#         # 5.4. Display Final Count
#         st.subheader("3. Final Result")
#         st.metric(
#             label="Corrected Colony Count", 
#             value=f"{corrected_count}",
#             delta=corrected_count - initial_count,
#             delta_color="normal"
#         )
        
#         st.info(f"Initial Model Count: {initial_count} | Additions: +{added_count} | Removals: -{removed_count}")
        
#         # Add a clear button for the user to reset their manual corrections
#         if st.button("Clear All Corrections"):
#             st.session_state["drawing_json"] = None
#             st.rerun()

#     else:
#         st.warning("The annotated image data is corrupt or empty. Please re-upload your image and run inference again.")
