import streamlit as st

# Import tf_keras instead of tensorflow.keras
import tf_keras as keras
import os
import numpy as np
from PIL import Image

st.title("Demo 🤖")
st.caption("*Deeper models take longer to predict.")
st.divider()


@st.cache_resource
def load_model():
    model_path = "models/152_50_0.0001_128/"
    
    # Check if the model directory exists
    if not os.path.exists(model_path):
        st.error(f"Model directory not found at {model_path}. Please ensure the model is present.")
        return None
    
    try:
        # Load the model using tf_keras
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


st.subheader("Model Configuration")
st.text("The highest performing is 152, 50, 0.0001, 128, which is demonstrated here.")

st.subheader("Test Input")

directory = "test"

# Check if the test directory exists
if not os.path.exists(directory):
    st.error(f"Test directory '{directory}' does not exist.")
    images = []
else:
    images = os.listdir(directory)

# Allow user to select an image source
option = st.radio(
    "Select Image Source:", ("Select from provided images", "Upload an image")
)

if option == "Select from provided images":
    if images:
        num_columns = 8
        col_count = 0
        for image in images:
            if col_count % num_columns == 0:
                columns = st.columns(num_columns)
            img_path = os.path.join(directory, image)
            if os.path.isfile(img_path):
                try:
                    columns[col_count % num_columns].image(
                        img_path, use_container_width=True, caption=image
                    )
                except Exception as e:
                    st.warning(f"Could not load image {image}: {e}")
            col_count += 1

        selected_image = st.selectbox("Select", images)
        uploaded_image = None
    else:
        st.warning(f"No images found in the '{directory}' directory.")
        selected_image = None
        uploaded_image = None
else:
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    selected_image = None

# Prediction Section
if (selected_image is not None or uploaded_image is not None):
    with st.spinner("Predicting ⌛"):
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
        
        model = load_model()
        if model is None:
            st.error("Model could not be loaded. Please check the logs for more details.")
        else:
            # Load and preprocess the image
            try:
                if uploaded_image is not None:
                    input_image = Image.open(uploaded_image).resize((224, 224)).convert("RGB")
                else:
                    img_path = os.path.join(directory, selected_image)
                    input_image = Image.open(img_path).resize((224, 224)).convert("RGB")
            except Exception as e:
                st.error(f"Error processing image: {e}")
                input_image = None

            if input_image:
                try:
                    # Display the image
                    st.image(input_image, use_container_width=True)

                    # Convert image to array and preprocess
                    resized_image_array = np.asarray(input_image)
                    resized_image_array = resized_image_array / 255.0  # Normalize if required
                    resized_image_array = np.expand_dims(resized_image_array, axis=0)  # Add batch dimension

                    # Make prediction
                    raw_tensor_prediction = model.predict(resized_image_array)
                    predicted_class_index = np.argmax(raw_tensor_prediction, axis=1)[0]
                    predicted_class = class_names[predicted_class_index]
                    confidence = raw_tensor_prediction[0][predicted_class_index]

                    # Display results
                    st.metric(label="Predicted Class", value=predicted_class)
                    st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")