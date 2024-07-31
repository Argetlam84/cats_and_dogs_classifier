import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tensorflow.keras.models import load_model

# Function to load Keras models
@st.cache_data
def load_keras_model(model_name):
    try:
        if model_name == "MobileNet":
            model = load_model('keras_mobilenet_model.keras')
        elif model_name == "NASNet":
            model = load_model('keras_nasnet_model.keras')
        else:
            model = None
        return model
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None

# Function to load PyTorch models
@st.cache_resource
def load_pytorch_model(model_name):
    try:
        if model_name == "MobileNet":
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
            model.load_state_dict(torch.load('pytorch_mobilenet_model.pth'))
        elif model_name == "SqueezeNet":
            model = models.squeezenet1_0(pretrained=False)
            model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
            model.num_classes = 2
            model.load_state_dict(torch.load('pytorch_squeezenet_model.pth'))
        else:
            model = None
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None

# Preprocessing function
def preprocess_image(image, model_name):
    if model_name in ["Keras MobileNet", "Keras NASNet"]:
        image = image.resize((224, 224))
        image = np.array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)
    return image

# Prediction function
def predict(image, model, model_name):
    if model_name.startswith("Keras"):
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
    else:
        with torch.no_grad():
            outputs = model(image)
            _, predicted_class = torch.max(outputs, 1)
            predicted_class = predicted_class.item()
    return predicted_class

# Main page function
def main():
    st.title("Welcome to the Image Classification Project")
    st.write("This project allows you to classify images using pre-trained models.")
    st.image("https://github.com/Argetlam84/cats_and_dogs_classifier/issues/1#issue-2440765272", caption="Image Classification Project", use_column_width=True)
    st.write("""
        ## Project Overview
        This project demonstrates image classification using Keras and PyTorch models.
        You can upload an image and select a model to classify the image as either a cat or a dog.
        
        ## Instructions
        1. Select the model type and specific model from the sidebar.
        2. Upload an image.
        3. Click the 'Predict' button to see the classification result.
        
        ## Models
        - **Keras Models**: MobileNet, NASNet
        - **PyTorch Models**: MobileNet, SqueezeNet
    """)

# Prediction page function
def prediction_page():
    st.title("Image Classification")

    model_type = st.sidebar.selectbox("Select Model Type", ("Keras", "PyTorch"))

    if model_type == "Keras":
        model_name = st.sidebar.selectbox("Select Keras Model", ("MobileNet", "NASNet"))
        model = load_keras_model(model_name)
    else:
        model_name = st.sidebar.selectbox("Select PyTorch Model", ("MobileNet", "SqueezeNet"))
        model = load_pytorch_model(model_name)

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        if st.button('Predict'):
            if model is not None:
                image = preprocess_image(image, f"{model_type} {model_name}")
                class_idx = predict(image, model, f"{model_type} {model_name}")

                class_labels = {0: "cat", 1: "dog"}
                class_name = class_labels[class_idx]
                st.balloons()
                st.success(f"Predicted class: {class_name}")
            else:
                st.error("Failed to load model. Please check the model files and try again.")

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ("Main", "Prediction"))

if page == "Main":
    main()
else:
    prediction_page()
