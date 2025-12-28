"""
Alzheimer's Disease MRI Classification Web Application

This Streamlit application provides a user-friendly interface for classifying
brain MRI scans into four stages of Alzheimer's disease using a fine-tuned
ResNet-18 deep learning model.

Model Performance:
    - Accuracy: 92.89%
    - F1-Score: 92.83%

Classes:
    0: Mild Impairment
    1: Moderate Impairment
    2: No Impairment
    3: Very Mild Impairment

Author: AI4Alzheimers Team
Date: December 2025
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Configure Streamlit page settings
st.set_page_config(page_title="Alzheimer MRI Classifier", layout="centered")
st.title("Alzheimer's MRI Classifier")

# ---- Load trained model ----
@st.cache_resource
def load_model():
    """
    Load the pre-trained ResNet-18 model for Alzheimer's classification.
    
    The model is cached by Streamlit to avoid reloading on each interaction.
    Uses CPU inference for compatibility across different deployment environments.
    
    Returns:
        torch.nn.Module: Loaded ResNet-18 model in evaluation mode with
                        modified final layer for 4-class classification.
    
    Model Architecture:
        - Base: ResNet-18 (pre-trained on ImageNet)
        - Modified: Final FC layer adjusted for 4 output classes
        - Fine-tuned on Alzheimer's MRI dataset
    """
    model = models.resnet18(pretrained=False)
    # Replace final fully connected layer for 4-class classification
    model.fc = nn.Linear(model.fc.in_features, 4)
    # Load trained weights from disk
    model.load_state_dict(torch.load("alz_resnet18.pt", map_location="cpu"))
    # Set to evaluation mode (disables dropout, batch norm training behavior)
    model.eval()
    return model

# Initialize model (cached for performance)
model = load_model()

# ---- Preprocessing (must match training) ----
# These transformations MUST exactly match the training preprocessing
# to ensure consistent model predictions
transform = transforms.Compose([
    # Resize all images to 224x224 (ResNet-18 input size)
    transforms.Resize((224, 224)),
    # Convert PIL Image to PyTorch tensor (values in [0, 1])
    transforms.ToTensor(),
    # Normalize using ImageNet statistics (model was pre-trained on ImageNet)
    # Mean and std values are standard for ImageNet-pretrained models
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # RGB channel means
        std=[0.229, 0.224, 0.225]     # RGB channel standard deviations
    )
])

# ---- Label mapping (must match training dataset) ----
# Class indices are determined by alphabetical folder order in training dataset
# Based on dataset.class_to_idx: {'Mild Impairment': 0, 'Moderate Impairment': 1, 
#                                  'No Impairment': 2, 'Very Mild Impairment': 3}
class_names = [
    "Mild Impairment",        # 0 - Early stage cognitive decline
    "Moderate Impairment",    # 1 - Significant cognitive impairment
    "No Impairment",          # 2 - Healthy brain
    "Very Mild Impairment"    # 3 - Minimal cognitive changes
]

# ---- User Interface ----
# File uploader widget for MRI images
uploaded = st.file_uploader("Upload MRI image", type=["jpg", "png", "jpeg"])

if uploaded:
    # Load and display the uploaded image
    image = Image.open(uploaded).convert("RGB")  # Ensure 3-channel RGB
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Preprocess the image for model input
    # unsqueeze(0) adds batch dimension: [C, H, W] -> [1, C, H, W]
    x = transform(image).unsqueeze(0)

    # Perform inference without gradient computation (saves memory)
    with torch.no_grad():
        # Forward pass: get raw model outputs (logits)
        logits = model(x)
        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits, dim=1)
        # Get predicted class (highest probability)
        pred = probs.argmax(dim=1).item()
        # Get confidence score for the predicted class
        confidence = probs[0][pred].item()

    # Display results with confidence threshold
    # Confidence < 60% suggests the image may not be a valid MRI
    if confidence < 0.6:
        st.warning("Image not recognized as a valid brain MRI.")
    else:
        st.subheader("Prediction")
        st.success(class_names[pred])
        st.write(f"Confidence: {confidence*100:.2f}%")

