# AI4Alzheimers

An AI-powered MRI classification system for detecting different stages of Alzheimer's disease using deep learning.

## Overview

This project uses a ResNet-18 model to classify brain MRI scans into four categories:
- No Impairment
- Very Mild Impairment
- Mild Impairment
- Moderate Impairment

## Project Structure

```
AI4Alzheimers/
├── app/                    # Streamlit web application
│   └── app.py             # Main app file
├── data/                   # Dataset (excluded from git)
│   └── Combined Dataset/
│       ├── train/
│       └── test/
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   ├── label_check.ipynb
│   └── train_model.ipynb
└── README.md
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd AI4Alzheimers
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

Run the training notebook:
```bash
jupyter notebook notebooks/train_model.ipynb
```

The trained model will be saved as `alz_resnet18.pt` in the notebooks directory. Copy it to the `app/` directory to use with the Streamlit app.

## Running the Web App

```bash
cd app
streamlit run app.py
```

Upload an MRI image to get a prediction with confidence score.

## Model Details

- Architecture: ResNet-18 (transfer learning)
- Input: 224x224 RGB images
- Output: 4 classes
- Training: Fine-tuned on custom Alzheimer's MRI dataset

## License

This project is for educational purposes.
