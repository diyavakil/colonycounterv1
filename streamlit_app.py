from streamlit_drawable_canvas import st_canvas
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# custom page title
st.set_page_config(page_title="Colony Counter v1")

# custom page icon
# from PIL import Image
# im = Image.open("App_Icon3.jpg") # cute squid
# st.set_page_config(page_icon=im)

# header
st.title("🧫 Colony Counter v1")

model = YOLO("weights.pt") # load weights

# Sidebar for correction tools
st.sidebar.header("Correction Tools")
mode = st.sidebar.radio("Mode", ["None", "➕ Add", "➖ Remove"])
removed = st.sidebar.number_input

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
        
        # draw only the cute little green bboxes
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
       
        # Count colonies
        colony_count = len(results[0].boxes.xyxy)

        # save results into session state
        st.session_state["img_annotated"] = img_annotated
        st.session_state["colony_count"] = colony_count

    # if we already ran inference, display the results
    if "img_annotated" in st.session_state:
        img_annotated = st.session_state["img_annotated"]
        colony_count = st.session_state["colony_count"]

        st.subheader("📊 Annotated Image & Corrections")

        # correction canvas directly on YOLO image 
        canvas_result = st_canvas(
            fill_color="rgba(0,255,0,0.3)" if mode == "➕ Add" else "rgba(255,0,0,0.3)", 
            stroke_width=2,
            stroke_color="green" if mode == "➕ Add" else "red", 
            background_image=Image.fromarray(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)), 
            update_streamlit=True, 
            height=img_annotated.shape[0], 
            width=img_annotated.shape[1], 
            drawing_mode="rect" if mode in ["➕ Add", "➖ Remove"] else "none", 
            key="correction_canvas" 
        )
        
        # # Add count text in bottom-right corner
        # text = f"Colonies: {colony_count}"
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 3 # text size
        # thickness = 5 # text thickness
        # text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        # text_x = img_annotated.shape[1] - text_size[0] - 10  # 10 px from right
        # text_y = img_annotated.shape[0] - 10  # 10 px from bottom
        # cv2.putText(img_annotated, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
        
        # st.image(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_column_width=True)
        
        # # save img to file
        # save_path = "annotated_streamlit.jpg"
        # cv2.imwrite(save_path, img_annotated)
        # st.success(f"Annotated image saved as {save_path}")
        # st.download_button(
        #     label="Download Annotated Image",
        #     data=open(save_path, "rb").read(),
        #     file_name="annotated_image.jpg",
        #     mime="image/jpeg"
        # )

    # # if we already ran inference, display the results
    # if "img_annotated" in st.session_state:
    #     img_annotated = st.session_state["img_annotated"]
    #     colony_count = st.session_state["colony_count"]
    
    #     st.image(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB),
    #              caption="Annotated Image", use_column_width=True)

        # # --- Human-in-the-loop correction DOES NOT WORK!!!!!
        # st.subheader("✏️ Review & Correct Colonies")

        # st.markdown(
        #     "Draw green rectangles on colonies the model missed. "
        #     "If the model added extra ones, enter how many to remove below."
        # )

        # canvas_result = st_canvas(
        #     fill_color="rgba(0, 255, 0, 0.3)",
        #     stroke_width=2,
        #     stroke_color="green",
        #     background_image=Image.open(save_path),
        #     update_streamlit=True,
        #     height=img_annotated.shape[0],
        #     width=img_annotated.shape[1],
        #     drawing_mode="rect",
        #     key="canvas"
        # )

        added = len(canvas_result.json_data["objects"]) if canvas_result.json_data and mode == "➕ Add" else 0 
        manually_removed = removed + (len(canvas_result.json_data["objects"]) if canvas_result.json_data and mode == "➖ Remove" else 0)
        
        corrected_count = colony_count + added - nanually_removed
        
        st.success(f"✅ Corrected Colony Count: {corrected_count}")

