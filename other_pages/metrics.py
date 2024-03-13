import streamlit as st 
import pandas as pd

data = pd.read_csv("grid_search_data.csv")

st.title("Metrics 📈")
st.divider()

st.header("Model-Specific Metrics")
DEPTH = st.selectbox("ResNet depth (layers)", [50,101,152])
EPOCHS = st.selectbox("Epochs", [10,25,40,50])
LEARNING_RATE = st.selectbox("Learning rate", [0.1, 0.001, 0.0001, 1e-7])
BATCH_SIZE = st.selectbox("Batch size", [16,32,64,128,256])

st.subheader("Accuracy & Loss")
st.text("Understand the model's learning journey")
st.image(f"models/{DEPTH}_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}/loss_and_accuracy_plot.png")

st.subheader("Validation Confusion Matrix")
st.text("Understand the model's prediction patterns")
st.image(f"models/{DEPTH}_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}/confusion_matrix.png")

st.subheader("More Metrics")

filtered_data = data[
    (data['DEPTH'] == DEPTH) &
    (data['EPOCHS'] == EPOCHS) &
    (data['LEARNING_RATE'] == LEARNING_RATE) &
    (data['BATCH_SIZE'] == BATCH_SIZE)
]

# Display the metrics
if not filtered_data.empty:
    st.write("Train Accuracy:", filtered_data['Train Accuracy'].values[0])
    st.write("Train Loss:", filtered_data['Train Loss'].values[0])
    st.write("Validation Accuracy:", filtered_data['Validation Accuracy'].values[0])
    st.write("Validation Loss:", filtered_data['Validation Loss'].values[0])
    st.write("Validation Precision:", filtered_data['Validation Precision'].values[0])
    st.write("Validation Recall:", filtered_data['Validation Recall'].values[0])
    st.write("Validation F1 Score:", filtered_data['Validation F1 Score'].values[0])
st.divider()

st.header("Performance of All Configurations")
st.dataframe(data=data)

# FIGURES ONCE TRAINING IS FINISHED