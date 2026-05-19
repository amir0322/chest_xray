# 🫁 Chest X-Ray AI Classifier Dashboard

An interactive web-based dashboard for chest X-ray classification using a trained DenseNet-121 deep learning model. Built with Streamlit and featuring a professional green, yellow, black, and white color scheme.

## 🎯 Features

- **Interactive Image Upload**: Upload chest X-ray images (PNG, JPG, JPEG)
- **Real-time Predictions**: Get instant AI-powered classification results
- **Confidence Scoring**: See prediction confidence with visual gauges
- **Detailed Analysis**: View probabilities for all 5 classes
- **Professional UI**: Custom-designed dashboard with intuitive navigation
- **Color-coded Results**: Easy visual interpretation with consistent theming
- **Model Information**: Learn about the model architecture and capabilities
- **Multiple Sections**:
  - 📊 Upload & Predict: Main prediction interface
  - 📈 Model Info: Technical details about the model
  - 📚 About: Information and disclaimer
  - ⚙️ Settings: Theme and configuration details

## 📋 Classification Classes

The model classifies chest X-rays into 5 categories:
1. **Normal** ✅ - Healthy chest X-rays
2. **COVID-19** ⚠️ - COVID-19 related findings
3. **Bacterial Pneumonia** ⚠️ - Bacterial pneumonia indicators
4. **Viral Pneumonia** ⚠️ - Viral pneumonia patterns
5. **Tuberculosis** 🚨 - Tuberculosis findings

## 🎨 Color Scheme

- **Green (#4CAF50)** - Primary accent, normal/healthy indicators
- **Yellow (#FFD700)** - Secondary accent, medium alerts
- **Black (#1a1a1a)** - Sidebar and dark elements
- **White** - Background and text for high contrast

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone/Navigate to the project directory:**
   ```bash
   cd "C:\Users\amirz\OneDrive\Desktop\xray"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Activate it
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard

```bash
streamlit run dashboard.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📦 Files

- `chest_xray_5class_densenet.keras` - Trained model file
- `dashboard.py` - Main Streamlit application
- `chestx_ray.ipynb` - Jupyter notebook with model training code
- `requirements.txt` - Python package dependencies
- `README.md` - This file

## 💡 Usage Guide

### Uploading an Image
1. Navigate to the "📊 Upload & Predict" tab
2. Click on the file uploader area
3. Select a chest X-ray image (PNG, JPG, JPEG)
4. The image will be processed and prediction will appear automatically

### Understanding Results
- **Result Card**: Shows the predicted class and confidence percentage
- **Confidence Gauge**: Visual representation of prediction confidence
- **All Class Probabilities**: Horizontal bar chart showing all predictions
- **Analysis Section**: Interpretation of confidence level

### Interpreting Confidence
- 🟢 **> 85%**: High confidence - Results are reliable
- 🟡 **70-85%**: Good confidence - Results are reasonably reliable
- 🟠 **< 70%**: Moderate confidence - Consider manual review

## ⚠️ Important Disclaimer

**This application is for educational and research purposes only.**

- ❌ **NOT** a substitute for professional medical diagnosis
- 🏥 **Always** consult qualified radiologists for diagnosis
- 👨‍⚕️ **Results require** validation by medical professionals
- 🚫 **DO NOT** use for critical care decisions without expert review

## 🔧 Technical Details

### Model Architecture
- **Base Model**: DenseNet-121 (pre-trained on ImageNet)
- **Transfer Learning**: Leverages pre-trained features
- **Custom Layers**: 
  - Global Average Pooling
  - Dense layer with 256 neurons (ReLU activation)
  - Batch Normalization
  - Dropout (40%)
  - Output layer with Softmax for 5 classes

### Training Details
- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Callbacks**: Early Stopping, Learning Rate Reduction
- **Input Size**: 224×224 pixels, RGB format

## 🎓 Dataset

The model was trained on a combined dataset including:
- COVID-19 and TB chest X-rays
- Normal and Pneumonia (Bacterial/Viral) samples
- Multiple public medical imaging sources
- Diverse patient populations for better generalization

## 🔧 Troubleshooting

### Port Already in Use
```bash
streamlit run dashboard.py --server.port 8502
```

### Model Not Found
Ensure `chest_xray_5class_densenet.keras` is in the same directory as `dashboard.py`

### Image Upload Issues
- Check file format (PNG, JPG, JPEG)
- Ensure file size is reasonable
- Try converting to different format if upload fails

### TensorFlow Issues
```bash
pip install --upgrade tensorflow
```

## 📊 Performance Metrics

The model achieves good performance on:
- Normal chest X-rays
- COVID-19 detection
- Pneumonia classification
- Tuberculosis identification

*Note: Actual performance may vary based on image quality and patient demographics*

## 🤝 Contributing

To improve the dashboard:
1. Modify `dashboard.py` for UI changes
2. Retrain the model using `chestx_ray.ipynb`
3. Update the color scheme in the CSS section

## 📄 License

Educational and research use only.

## 👥 Support

For issues or questions:
1. Check the About tab for information
2. Review the Model Info tab for technical details
3. Ensure all requirements are installed correctly
4. Verify the model file exists in the project directory

---

**Made with ❤️ for Medical AI Research**
