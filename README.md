# 🚦 Traffic Sign Recognition System

A professional-grade Streamlit application for real-time, batch, and educational recognition of traffic signs using Deep Learning.  
Built with modern UI, features interactive visualizations, and supports both single and batch image analysis, as well as webcam-based detection.

---

## 📜 Project Overview

The **Traffic Sign Recognition System** is a deep learning project focused on classifying German traffic signs from images. It leverages a custom-trained CNN model, built and deployed with [Streamlit](https://streamlit.io/), and offers an intuitive UI for users to interact, test, and learn about traffic signs.

![Project Showcase](traffic_sign_recognition_logo.png)

---

## 📝 Description

- **Dataset Used:** [GTSRB (Kaggle)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Goal:** Classify traffic signs based on their image using a convolutional neural network (CNN)
- **Preprocessing:** Image resizing, normalization, and data augmentation
- **Model:** Custom CNN trained to recognize multiple traffic sign classes
- **Evaluation:** Accuracy, confusion matrix, and top-k prediction confidences

---

## 🛠️ Tools & Libraries

| Python | Keras | TensorFlow | OpenCV | Streamlit | Matplotlib | Pandas |
|--------|-------|------------|--------|-----------|------------|--------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | ![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white) | ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) | ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white) | ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white) | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=matplotlib&logoColor=white) | ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) |

---

## 📚 Covered Topics

- **Computer Vision (CNN)**
- **Multi-class Image Classification**
- **Real-time Detection with Webcam**
- **Batch Processing**
- **Interactive Data Visualization**
- **Streamlit UI/UX Design**

---

## ✨ Features

- **Single Image Prediction**: Upload an image and get immediate traffic sign classification with top-5 confidence scores.
- **Batch Processing**: Upload multiple images and get a detailed report with downloadable results.
- **Real-time Detection**: Use your webcam for live sign detection and recognition.
- **Traffic Sign Guide**: Learn about every supported traffic sign with visual cues and explanations.
- **Attractive, Responsive UI**: Professional dark/light theme, custom logo, clear visual feedback.
- **Performance Visualization**: See confidence levels and prediction summaries with interactive charts.
- **Educational Mode**: Explore sign categories, meanings, and test your knowledge.

---

## 🚀 Bonus

- **Data Augmentation**: Improves model generalization and accuracy.
- **Compare Models**: Evaluate custom CNN vs. pre-trained architectures (e.g., MobileNet).
- **Export Results**: Download batch prediction results as CSV for further analysis.

---

## 🖥️ App Preview

![App Screenshot](traffic_sign_recognition_logo.png)

---

## 📦 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Model

Place your trained model (`traffic_sign_cnn.h5`) in the project directory or update the path in the code.

### 4. Run the App

```bash
streamlit run traffic_sign_recog.py
```

---

## 🌐 Dataset

- **GTSRB - German Traffic Sign Recognition Benchmark**  
  [Kaggle Dataset Link](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

---

## 📈 Results & Evaluation

- **Accuracy & Confusion Matrix**: Evaluated on GTSRB test set.
- **Top-5 Predictions**: Visualized in charts for each prediction.
- **Batch results**: Downloadable as CSV for further analysis.

---

## 👨‍💻 Author

**Ummee Habiba**  
[GitHub Profile](https://github.com/Ummee-Habiba)
---

## 🏷️ License

This project is for educational and research purposes.

---

## 🏆 Bonus Ideas

- Add more real-world images for testing robustness.
- Integrate with mobile app or Raspberry Pi for on-the-road testing.
- Extend to other countries' traffic signs.

---

## 📷 Sample Output

> *"Upload an image, and instantly know the traffic sign!"*

---

## 🙌 Acknowledgments

- GTSRB Dataset by Benchmark Authors
- [Kaggle](https://www.kaggle.com/)
- Open Source Libraries

---

![Task Description](image2)
