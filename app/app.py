import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Alzheimer MRI Classifier", layout="centered")
st.title("Alzheimer’s MRI Classifier")

# ---- Load trained model ----
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load("alz_resnet18.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---- Preprocessing (must match training) ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---- Label mapping (LOCKED) ----
class_names = [
    "Mild Demented",        # 0
    "Moderate Demented",    # 1
    "Non-Demented",         # 2
    "Very Mild Demented"    # 3
]

# ---- UI ----
uploaded = st.file_uploader("Upload MRI image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0][pred].item()


    if confidence < 0.6:
        st.warning("Image not recognized as a valid brain MRI.")
    else:
        st.subheader("Prediction")
        st.success(class_names[pred])
        st.write(f"Confidence: {confidence*100:.2f}%")

