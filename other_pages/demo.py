import streamlit as st
from st_pages import Page, show_pages, add_page_title

import tensorflow as tf
import os
import numpy as np
from PIL import Image

st.title("Demo 🤖")
st.caption("*Deeper models take longer to predict.")
st.divider()

st.subheader("Model Configuration")
st.text("The highest performing is 152, 50, 0.0001, 128, which is demonstrated here.")
with st.spinner("Loading model ⌛"):
    model = tf.keras.models.load_model(f"models/152_50_0.0001_128/")
    st.success("Done!")

st.subheader("Test Input")

directory = "test"
images = os.listdir(directory)

# Allow user to select an image
option = st.radio(
    "Select Image Source:", ("Select from provided images", "Upload an image")
)

if option == "Select from provided images":
    num_columns = 8
    col_count = 0
    for image in images:
        if col_count % num_columns == 0:
            columns = st.columns(num_columns)
        img_path = os.path.join(directory, image)
        columns[col_count % num_columns].image(
            img_path, use_column_width=True, caption=image
        )
        col_count += 1

    selected_image = st.selectbox("Select", images)
    uploaded_image = None
else:
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    selected_image = None

# Prediction
if selected_image is not None or uploaded_image is not None:
    with st.status("Predicting ⌛"):
        class_names = [
            "Apple___Apple_scab",
            "Apple___Black_rot",
            "Apple___Cedar_apple_rust",
            "Apple___healthy",
            "Blueberry___healthy",
            "Cherry_(including_sour)___Powdery_mildew",
            "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy",
            "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Grape___healthy",
            "Orange___Haunglongbing_(Citrus_greening)",
            "Peach___Bacterial_spot",
            "Peach___healthy",
            "Pepper,_bell___Bacterial_spot",
            "Pepper,_bell___healthy",
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Raspberry___healthy",
            "Soybean___healthy",
            "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch",
            "Strawberry___healthy",
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy",
        ]
        st.write("Loaded model ✅")
        if uploaded_image is not None:
            input_image = Image.open(uploaded_image).resize((224, 224)).convert("RGB")
        else:
            input_image = (
                Image.open(os.path.join(directory, selected_image))
                .resize((224, 224))
                .convert("RGB")
            )
        st.write("Loaded input ✅")
        resized_image_array = np.expand_dims(np.asarray(input_image), axis=0)
        st.write("Resized input ✅")
        raw_tensor_prediction = model.predict(resized_image_array)
        st.write("Extracted features ✅")
        predicted_class_index = np.argmax(raw_tensor_prediction, axis=1)
        predicted_class = class_names[predicted_class_index[0]]
        confidence = raw_tensor_prediction[0][predicted_class_index[0]]
        st.write("Predicted class ✅")
        st.image(input_image, use_column_width=True)
        st.metric(label="Predicted Class", value=predicted_class)
        st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")
