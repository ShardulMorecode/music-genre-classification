# pip install numpy pandas librosa scikit-learn tensorflow streamlit matplotlib seaborn

# Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import librosa

# Load the 30-second feature CSV from the specified path
features_30_sec = pd.read_csv(r"D:\MP 3\archive (1)\Data\features_30_sec.csv")

# Inspect the dataset
print("Dataset Overview:")
print(features_30_sec.head())
print(features_30_sec.info())

# Separate features and labels
X = features_30_sec.drop(columns=['filename', 'label']).values  # Drop the filename and label columns
y = features_30_sec['label'].values  # Use only the label column for y

# Encode genre labels into numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encode the labels (since this is a classification problem)
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Build the Neural Network model
model = Sequential()

# Input layer
model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(BatchNormalization())  # Added batch normalization
model.add(Dropout(0.3))  # Adding dropout to prevent overfitting

# Hidden layers
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())  # Added batch normalization
model.add(Dropout(0.3))  # Added dropout

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())  # Added batch normalization
model.add(Dropout(0.3))  # Added dropout

# Output layer (Number of classes = number of unique genres)
model.add(Dense(y_train_encoded.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train_scaled, y_train_encoded, 
                    epochs=50, 
                    batch_size=32, 
                    validation_data=(X_test_scaled, y_test_encoded),
                    callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Optional: Plot training & validation accuracy and loss
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()

# Make predictions on the test set
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

# Decode the true labels using the same label encoder
y_test_decoded = label_encoder.inverse_transform(y_test)

# Decode the predictions back to original genre names
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Classification report
print("Classification Report:")
print(classification_report(y_test_decoded, y_pred_decoded))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_decoded, y_pred_decoded)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Save the model to a file
model.save("genre_classification_model.h5")
print("Model saved as 'genre_classification_model.h5'")

# Save LabelEncoder and StandardScaler
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)












