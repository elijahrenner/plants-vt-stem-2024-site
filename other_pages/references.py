import streamlit as st

st.title("References")

references = [
    'Desarda, Akash. "Bias-Variance & Precision-Recall Trade-offs: How to aim for the sweet spot." Towards Data Science, Medium, [link](https://towardsdatascience.com/tradeoffs-how-to-aim-for-the-sweet-spot-c20b40d5e6b6)',
    'Singh, Seema. "Understanding the Bias-Variance Tradeoff." Towards Data Science, Medium, [link](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)',
    "Pathogens, precipitation and produce prices. Nature Climate Change, vol. 11, no. 8, 2021, pp. 635–635.",
    'Rezvantalab, A., Safigholi, H., & Karimijeshni, S. "Dermatologist Level Dermoscopy Skin Cancer Classification Using Different Deep Learning Convolutional Neural Networks Algorithms."',
    'Hauptmann, A., & Adler, J. "On the unreasonable effectiveness of CNNs."',
    'Hughes, D. P., & Salathé, M. "An open access repository of images on plant health to enable the development of mobile disease diagnostics."',
    'El-Kereamy, A. et al. "Using Deep Learning for Image-Based Plant Disease Detection." Frontiers in Plant Science, vol. 7, 2016, p. 1419, [link](www.frontiersin.org)',
    'Nabi, Javaid. "Hyper-parameter Tuning Techniques in Deep Learning." Towards Data Science, Medium, [link](https://towardsdatascience.com/hyper-parameter-tuning-techniques-in-deep-learning-4dad592c63c8)',
    'Bergstra, J., Ca, J. B., & Ca, Y. B. "Random Search for Hyper-Parameter Optimization." Journal of Machine Learning Research, vol. 13, 2012, pp. 281–305.',
    '"Lecture 14: Hyperparameter Optimization." CS4787-Principles of Large-Scale Machine Learning Systems.',
    'Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. "Algorithms for Hyper-Parameter Optimization."',
    'Deng, J. et al. "ImageNet: A Large-Scale Hierarchical Image Database."',
    'Feurer, M., & Hutter, F. "Hyperparameter Optimization." doi:10.1007/978-3-030-05318-5_1.',
    '"New Plant Diseases Dataset." Kaggle, [link](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)',
    'He, K., Zhang, X., Ren, S., & Sun, J. "Deep Residual Learning for Image Recognition."',
]

for ref in references:
    st.markdown(ref)
