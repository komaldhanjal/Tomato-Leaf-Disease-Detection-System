# Tomato-Leaf-Disease-Detection-System

This repository contains an end-to-end Deep Learning project designed to classify 10 different types of tomato leaf diseases using a Convolutional Neural Network (CNN). The project includes the full model training pipeline and a user-friendly Streamlit web application for real-time diagnosis.

Features:

Deep Learning Architecture: A Sequential CNN model with multiple Convolutional, MaxPooling, and Dropout layers for robust feature extraction and classification.

High Performance: Trained on a dataset of 10,000 images with automated data augmentation (rotation, zoom, horizontal flip).

Interactive Web App: A Streamlit interface that allows users to upload an image of a leaf and receive an instant prediction of the disease type.

Automated Preprocessing: Real-time image resizing and normalization to match the model's input requirements ($224 \times 224$ pixels).

Dataset Information
The model is trained on tomato leaf images. You can download the primary dataset for this project from Kaggle:

Dataset Link: Tomato Leaf Disease Dataset (Kaggle)

Classes Identified: The system classifies leaves into categories such as Healthy, Early Blight, and Late Blight.

Dataset & Classes:
The model can identify the following 10 categories:

Bacterial Spot

Early Blight

Late Blight

Leaf Mold

Septoria Leaf Spot

Spider Mites (Two-spotted spider mite)

Target Spot

Yellow Leaf Curl Virus

Tech Stack:
Frameworks: TensorFlow, Keras

Data Processing: NumPy, ImageDataGenerator

Web App: Streamlit

Visualization: Matplotlib

Image Handling: PIL (Python Imaging Library)

How to Use
Launch the app via Streamlit.

Upload a .jpg, .jpeg, or .png image of a tomato leaf.

Click the "Predict" button.

The system will display the image and the predicted disease category.



Mosaic Virus

Healthy
