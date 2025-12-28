# AI4Alzheimers

An AI-powered MRI classification system for detecting different stages of Alzheimer's disease using deep learning.

## 📊 Model Performance

- **Accuracy**: 92.89%
- **F1-Score**: 92.83%
- **Architecture**: ResNet-18 (Fine-tuned)
- **Dataset**: 11,519 brain MRI images (10,240 train, 1,279 test)

## Overview

This project uses a pre-trained ResNet-18 model, fine-tuned on brain MRI scans, to classify images into four stages of Alzheimer's disease:

| Class                    | Description                         | Test Accuracy |
| ------------------------ | ----------------------------------- | ------------- |
| **No Impairment**        | Healthy brain, no cognitive decline | 97.50%        |
| **Very Mild Impairment** | Minimal cognitive changes           | 86.83%        |
| **Mild Impairment**      | Early stage cognitive decline       | 91.06%        |
| **Moderate Impairment**  | Significant cognitive impairment    | 100.00%       |

## Features

✅ **Production-ready Jupyter notebook** with evaluation metrics  
✅ **Interactive web interface** using Streamlit  
✅ **Relative paths** for reproducibility across environments  
✅ **Confusion matrix visualization** saved to outputs/  
✅ **Training safety guard** prevents accidental model retraining  
✅ **Comprehensive documentation** with docstrings and comments

## Project Structure

```
AI4Alzheimers/
├── app/                          # Streamlit web application
│   ├── app.py                   # Main application with UI and inference
│   └── alz_resnet18.pt          # Trained model weights (42.72 MB)
├── data/                         # Dataset (excluded from git)
│   └── Combined Dataset/
│       ├── train/               # Training images (10,240 images)
│       │   ├── Mild Impairment/
│       │   ├── Moderate Impairment/
│       │   ├── No Impairment/
│       │   └── Very Mild Impairment/
│       └── test/                # Test images (1,279 images)
│           ├── Mild Impairment/
│           ├── Moderate Impairment/
│           ├── No Impairment/
│           └── Very Mild Impairment/
├── notebooks/                              # Jupyter notebooks for development
│   ├── 01_eda.ipynb                       # Exploratory data analysis
│   ├── 02_preprocessing.ipynb             # Data preprocessing steps
│   ├── 03_baseline.ipynb                  # Baseline model experiments
│   ├── label_check.ipynb                  # Dataset label verification
│   ├── train_model.ipynb                  # Original training notebook
│   └── Alzheimers_MRI_Classification_Final.ipynb  # Production notebook with evaluation
├── outputs/                      # Generated outputs
│   └── confusion_matrix.png     # Visualization of model performance
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
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

### Using the Production Notebook (Recommended)

```bash
jupyter notebook notebooks/Alzheimers_MRI_Classification_Final.ipynb
```

**The notebook will:**

- ✅ Load the existing trained model by default (no retraining)
- ✅ Evaluate on test set and display metrics
- ✅ Generate confusion matrix visualization
- ✅ Save results to `outputs/confusion_matrix.png`

**To retrain from scratch** (not recommended unless needed):

1. Open the notebook
2. Set `TRAIN = True` in the model loading cell
3. Run all cells (⚠️ will overwrite existing model)

### Model Training Details

- **Base Model**: ResNet-18 (pre-trained on ImageNet)
- **Fine-tuning Strategy**: Last 2 blocks + final FC layer
- **Optimizer**: Adam (lr=1e-5)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 8 (training), 16 (evaluation)
- **Epochs**: 5
- **Data Augmentation**: Resize (224x224), Normalize (ImageNet stats)

## Running the Web App

1. Navigate to the app directory:

```bash
cd app
```

2. Start the Streamlit server:

```bash
streamlit run app.py
```

3. The app will open in your browser at `http://localhost:8501`

4. **How to use:**
   - Click "Browse files" to upload a brain MRI image (jpg, png, jpeg)
   - The model will classify the image and display:
     - Predicted Alzheimer's stage
     - Confidence score (%)
   - Images with confidence < 60% are flagged as potentially invalid

### Example Usage

```python
# The app automatically:
# 1. Loads the trained model (cached for performance)
# 2. Preprocesses uploaded images (resize, normalize)
# 3. Runs inference on CPU
# 4. Displays results with confidence threshold
```

## Model Details

### Architecture

- **Base**: ResNet-18 (18 deep convolutional layers)
- **Pre-training**: ImageNet (1.2M images, 1000 classes)
- **Modification**: Final FC layer replaced (512 → 4 classes)
- **Fine-tuning**: Layers 3, 4, and FC (remaining layers frozen)

### Input/Output Specifications

- **Input Shape**: (224, 224, 3) - RGB images
- **Preprocessing**:
  - Resize to 224×224
  - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Output**: 4-class probability distribution
- **Inference Time**: ~50-100ms per image (CPU)

### Training Dataset

- **Total Images**: 11,519 brain MRI scans
- **Split**: 88.9% train (10,240), 11.1% test (1,279)
- **Class Distribution** (train/test):
  - No Impairment: 2,560 / 640
  - Very Mild: 1,792 / 448
  - Mild: 716 / 179
  - Moderate: 48 / 12

## Tech Stack

### Core Frameworks

- **PyTorch** (2.9.1) - Deep learning framework
- **TorchVision** (0.24.1) - Computer vision utilities
- **Streamlit** (1.52.2) - Web application framework

### Data Science

- **scikit-learn** (1.3+) - Metrics (accuracy, F1, confusion matrix)
- **Matplotlib** (3.7+) - Plotting confusion matrix
- **Seaborn** (0.12+) - Statistical data visualization
- **Pillow** (12.0) - Image processing

### Development

- **Jupyter** (1.1.1) - Interactive notebooks
- **Python** (3.12.6) - Programming language

## Results & Evaluation

The model's performance is documented in `notebooks/train_model_FINAL.ipynb`:

```
📊 MODEL EVALUATION RESULTS
============================================================
✅ Accuracy: 92.89%
✅ F1-Score (weighted): 92.83%

🎯 PER-CLASS PERFORMANCE:
- Mild Impairment:      91.06% (163/179 correct)
- Moderate Impairment: 100.00% (12/12 correct)
- No Impairment:        97.50% (624/640 correct)
- Very Mild Impairment: 86.83% (389/448 correct)
```

**Confusion Matrix**: Available in `outputs/confusion_matrix.png`

## Repository

- **GitHub**: https://github.com/Anish-Kulkarni1/AI4Alzheimers
- **Model File**: Included in repo via Git LFS (alz_resnet18.pt, 42.72 MB)

## Contributing

This project was developed for educational purposes to demonstrate:

- Transfer learning with ResNet-18
- Medical image classification
- Production-ready ML pipeline
- Interactive ML deployment with Streamlit

## Acknowledgments

- **Dataset**: Brain MRI Images for Alzheimer's Disease Classification
- **Base Model**: ResNet-18 from PyTorch Model Zoo
- **Framework**: PyTorch, Streamlit

## License

This project is for educational purposes.

---

**Note**: This model is for educational/research purposes only and should not be used for medical diagnosis. Always consult healthcare professionals for medical advice.
