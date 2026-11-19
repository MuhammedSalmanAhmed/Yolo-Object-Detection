
import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(
    page_title="Duck Hunt Object Detector",
    page_icon="🦆",
    layout="wide"
)

st.title("🦆 Duck Hunt Object Detector")

st.write(
    "Upload an image to detect dogs and different duck categories "
    "using our trained YOLOv8 model."
)

@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "best.pt"
    )
    return YOLO(model_path)

model = load_model()

st.subheader("Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Objects", type="primary"):

        with st.spinner("Detecting objects..."):

            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=640,
                verbose=False
            )

            result = results[0]

        st.subheader("Detection Results")

        annotated_image = result.plot()

        annotated_image = Image.fromarray(
            annotated_image[:, :, ::-1]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                annotated_image,
                caption="Detected Objects",
                use_container_width=True
            )

        detections = []

        for box in result.boxes:

            class_id = int(box.cls[0])
            score = float(box.conf[0])

            detections.append({
                "Object": result.names[class_id],
                "Confidence": f"{score * 100:.2f}%"
            })

        with col2:

            st.metric(
                "Total Objects Detected",
                len(detections)
            )

            if detections:

                st.success("Objects detected successfully!")

                st.dataframe(
                    pd.DataFrame(detections),
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.warning("No objects detected. Try lowering the confidence threshold.")

st.divider()

st.subheader("Supported Classes")

st.write(
    "🐕 Dog | 🦆 Dead Duck | 🦆 Shot Duck | 🦆 Flying Duck"
)

st.caption("Powered by YOLOv8n and Ultralytics.")
