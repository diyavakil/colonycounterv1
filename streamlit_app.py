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
        
        # draw only the little green bboxes
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
       
        # count colonies
        colony_count = len(results[0].boxes.xyxy)
        
        # add count in bottom-right corner
        text = f"Colonies: {colony_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3 # text size
        thickness = 5 # text thickness
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = img_annotated.shape[1] - text_size[0] - 10  # 10 px from right
        text_y = img_annotated.shape[0] - 10  # 10 px from bottom
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
