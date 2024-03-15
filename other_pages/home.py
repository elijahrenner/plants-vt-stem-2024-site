import streamlit as st

st.title("plants-vt-stem-2024 👨‍🌾")
st.subheader("CS-004 | Elijah Renner")
st.image("graphic.png")
st.divider()

st.header("Abstract")
st.markdown(
    "The novel 2015 paper Deep Residual Learning for Image Recognition introduced skip connections, solving the issue of vanishing gradients in deep neural networks. Since then, deeper neural networks have outperformed shallow networks on large multiclass datasets like ImageNET. Although effective, training deep neural networks involves meticulously tweaking hyperparameters (variables that control the “learning” process). Even if an experimenter tries numerous configurations, they aren’t sure if they’ve found the optimal configuration. We solve this issue by implementing a comprehensive Cartesian product search of 240 unique hyperparameter configurations of the ResNet architecture. The search isolates the effects of four hyperparameters: model depth, epochs, learning rate, and batch size. The PlantVillage dataset containing 38 classes is used to evaluate model performance when classifying plant images. We record accuracy, loss, precision, recall, and F1 score. The highest performing model, ResNet 152, achieves a validation accuracy of 99.038% and a validation loss of 0.040 when trained over 50 epochs with a learning rate of 0.0001 and a batch size of 128. Our results demonstrate that deep neural networks, when adjusted correctly, are trainable to perform complex tasks involving multiple classes. Furthermore, the model has important implications for detecting diseases, which can cause significant revenue losses in agriculture."
)

st.header("Poster")
st.image("Vermont STEM Fair 2024.png")
