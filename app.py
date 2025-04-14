import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle  # Use pickle to load the label encoder and scaler
import librosa

# Load the trained model
model = load_model('genre_classification_model.h5')

# Load the LabelEncoder and StandardScaler using pickle
with open(r"D:\MP 3\label_encoder.pkl", 'rb') as f:
    label_encoder = pickle.load(f)

with open(r"D:\MP 3\scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Function to predict genre
def predict_genre(features):
    # Feature scaling
    features_scaled = scaler.transform([features])  # Use the loaded scaler to transform features

    # Predict using the trained model
    prediction = model.predict(features_scaled)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)[0]

    return predicted_label

# Streamlit application setup
st.title("Music Genre Classification")
# Load and display the logo
logo_path = r"D:\MP 3\logo.jpg"  
st.image(logo_path, use_column_width=True)
st.write("Upload an audio file and we'll predict its genre!")

# File upload
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Process the uploaded audio file with librosa
    audio, sr = librosa.load(uploaded_file, sr=None)

    # Extract features (adjust according to your feature extraction method)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=58)  # Change n_mfcc to match your training data
    features_mean = np.mean(mfccs, axis=1)  # Take the mean of the MFCCs

    # Display the audio player in Streamlit
    st.audio(uploaded_file)

    # Predict the genre
    predicted_genre = predict_genre(features_mean)

    # Show the predicted genre
    st.success(f"Predicted Genre: {predicted_genre}")

# Optionally, display model evaluation or confusion matrix if needed
if st.button("Show Confusion Matrix"):
    st.write("Confusion Matrix will be generated here.")
