import streamlit as st

# custom page title
st.set_page_config(page_title="Colony Counter v1")

# custom page icon
# from PIL import Image
# im = Image.open("App_Icon3.jpg") # cute squid
# st.set_page_config(page_icon=im)

# header
st.title("🧫 Colony Counter v1")

#app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

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
        
        # draw only the cute little green bboxes
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
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

