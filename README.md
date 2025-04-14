# 🎧 Music Genre Classification with Deep Learning 🎶

Welcome to the **Music Genre Classification** project! This deep learning-powered web app can predict the genre of a song from an uploaded audio file using MFCC features and a neural network. Built with **Streamlit** for a smooth user experience and trained using **Keras** and **TensorFlow**.

## 🔥 Demo Screenshots

### 🎧 Upload Interface
![Upload Interface](https://github.com/ShardulMorecode/music-genre-classification/blob/main/archive%20(1)/Data/Screenshot%202025-04-14%20211140.png)

### 🎵 Prediction Result
![Prediction Result](https://github.com/ShardulMorecode/music-genre-classification/blob/main/archive%20(1)/Data/Screenshot%202025-04-14%20211221.png)

---

## 🚀 Features
- 🎼 Classifies music into multiple genres using a trained model
- 🧠 Built with a deep neural network using Keras
- 🎧 MFCC-based audio feature extraction with `librosa`
- 📊 Includes a confusion matrix and classification report for evaluation
- 🌐 User-friendly Streamlit interface

---

## 🗂️ Project Structure

```
MP 3/
│
├── Data/
│   ├── genres_original/
│   ├── images_original/
│   ├── features_3_sec.csv
│   └── features_30_sec.csv
│
├── hyperband_tuning/
│
├── app.py                      # 🎯 Streamlit web app for prediction
├── training.py                 # 🏋️ Training script for the model
├── genre_classification_model.h5
├── genre_classification_model.keras
├── label_encoder.pkl
├── scaler.pkl
├── logo.jpg / logo.png
└── archive (1).zip
```

---

## 🧠 Model Training Workflow

The model is trained using **features_30_sec.csv** and includes:

- 🔹 StandardScaler for feature normalization
- 🔹 LabelEncoder for genre labels
- 🔹 3-layer Dense Neural Network with dropout and batch normalization
- 🔹 Early stopping to avoid overfitting

```python
# Summary of the model architecture
Dense(128) → BatchNorm → Dropout
Dense(64)  → BatchNorm → Dropout
Dense(32)  → BatchNorm → Dropout
Dense(#classes, softmax)
```

✅ Final Model saved as `genre_classification_model.h5`  
✅ LabelEncoder and Scaler saved as `.pkl` files

---

## 🧪 Evaluation Metrics

- ✔️ **Classification Report**
- ✔️ **Confusion Matrix**
- ✔️ **Accuracy and Loss Plots**

These can be generated post-training using built-in visualizations and `scikit-learn`.

---

## 🌐 Web App (`app.py`)

The `Streamlit` app allows users to upload audio files (`.wav`, `.mp3`, `.ogg`) and predicts the genre in real-time.

### How it Works:
1. Upload an audio file.
2. MFCC features are extracted using `librosa`.
3. Features are scaled and passed to the model.
4. Predicted genre is displayed instantly! 🎉

---

## 📦 Setup Instructions

### 🔧 Install Dependencies
```bash
pip install numpy pandas librosa scikit-learn tensorflow streamlit matplotlib seaborn
```

### ▶️ Run the App
```bash
streamlit run app.py
```

---

## 🖼️ Sample Interface

![App Screenshot](https://github.com/yourusername/music-genre-classification/blob/main/logo.jpg)  
_Replace this link with a hosted image or use logo.png locally_

---

## 🤖 Tech Stack

- Python 🐍
- TensorFlow / Keras
- Scikit-learn
- Librosa 🎵
- Streamlit 🖥️
- Matplotlib & Seaborn

---

## 👨‍💻 Author

**Shardul More**  
📧 shardulmore777@gmail.com  
🎓 Final Year Student, Sanjay Ghodawat University

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⭐️ Show Your Support

If you like this project, consider giving it a ⭐ on GitHub!  
It helps me grow and keep building cool projects 💪

