# 🌿 LeafScan: Advanced Plant Disease Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://leafscan-57gequp3qt3evqnenpnytv.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Live Demo
**Try the application**: [LeafScan Web App](https://leafscan-57gequp3qt3evqnenpnytv.streamlit.app/)

---

## 🌟 Overview

**LeafScan** is a state-of-the-art AI-powered plant disease detection system designed to revolutionize agricultural diagnostics. By leveraging deep learning and computer vision, it provides instant, accurate identification of plant diseases from leaf images, enabling farmers and researchers to take timely action.

### 🎯 Problem Statement
Agriculture faces significant challenges with plant diseases causing:
- **20-40% global crop losses** annually
- **$220 billion economic impact** worldwide
- **Limited access** to plant pathology expertise
- **Delayed diagnosis** leading to disease spread

### 💡 Our Solution
LeafScan addresses these challenges by providing:
- **Disease detection** with 91% accuracy
- **38+ disease categories** across major crops
- **User-friendly web interface** accessible anywhere

---

## ✨ Key Features

### 🔬 Advanced AI Technology
- **Deep Learning Model**: MobileNetV2 with custom architecture
- **Transfer Learning**: Pre-trained on ImageNet for robust feature extraction
- **Focal Loss Function**: Handles class imbalance effectively
- **Data Augmentation**: Comprehensive preprocessing pipeline

### 🌐 Web Application
- **Streamlit Interface**: Intuitive drag-and-drop functionality
- **Real-time Processing**: Instant predictions with confidence scores
- **Mobile Responsive**: Works seamlessly on all devices
- **Error Handling**: Robust validation for image inputs

### 📊 Comprehensive Analytics
- **Detailed Performance Metrics**: Precision, recall, F1-score analysis
- **Confusion Matrix Visualization**: Clear model performance insights
- **Training History**: Complete model development tracking
- **Class Distribution Analysis**: Dataset balance visualization

---

## 🏗️ Technical Architecture

### 🧠 Model Architecture
```
Input Layer (224x224x3)
    ↓
MobileNetV2 Base (Pre-trained)
    ↓
Global Average Pooling
    ↓
Dense Layer (128 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Output Layer (38 classes, Softmax)
```

### 📊 Training Configuration
- **Framework**: TensorFlow 2.x / Keras
- **Loss Function**: Focal Loss (handles class imbalance)
- **Batch Size**: 32
- **Input Size**: 224x224x3
- **Epochs**: 10
- **Data Augmentation**: Rotation, flip, zoom, shift

### 🔧 Technology Stack
- **Backend**: Python, TensorFlow, Keras
- **Frontend**: Streamlit
- **Data Analysis**: NumPy
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Streamlit Cloud

---

## 📈 Performance Metrics

### 🎯 Overall Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 91% |
| **Macro Average F1** | 89% |
| **Weighted Average F1** | 91% |
| **Precision** | 92% |
| **Recall** | 91% |

### 🏆 Top Performing Classes
| Disease Category | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| Corn Common Rust | 0.97 | 1.00 | 0.98 |
| Squash Powdery Mildew | 1.00 | 0.99 | 1.00 |
| Strawberry Leaf Scorch | 1.00 | 0.95 | 0.98 |
| Apple Black Rot | 0.98 | 0.92 | 0.95 |
| Cherry Powdery Mildew | 1.00 | 0.95 | 0.98 |

### 📊 Model Training Analysis
- **Training Accuracy**: Improved from 69.6% to 83.1%
- **Validation Accuracy**: Achieved 91.2% peak performance
- **Loss Convergence**: Validation loss stabilized at ~0.037
- **Overfitting Control**: Effective regularization with dropout

### 🎨 Visualization Insights
1. **Training Curves**: Show steady improvement with good generalization
2. **Confusion Matrix**: Reveals strong performance across most classes
3. **Class Distribution**: Balanced dataset with appropriate weighting

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### 🔧 Local Development Setup

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/leafscan.git
cd leafscan
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Model (if not included)**
```bash
# Model should be in ./model/ directory
# Contact maintainer if model files are missing
```

### 🚀 Quick Start
```bash
# Run the Streamlit app
streamlit run app/streamlit_app.py

# Or run predictions via CLI
python src/predict.py path/to/image.jpg
```

---

## 📖 Usage

### 🌐 Web Application
1. **Access the App**: Visit [LeafScan Web App](https://leafscan-57gequp3qt3evqnenpnytv.streamlit.app/)
2. **Upload Image**: Drag and drop or browse for leaf image
3. **Get Prediction**: Instant disease classification with confidence score
4. **View Results**: Detailed analysis with recommendations

### 💻 Command Line Interface
```bash
# Single image prediction
python src/predict.py image.jpg

# Batch processing
python src/batch_predict.py images_folder/

# Model evaluation
python src/evaluate.py test_dataset/
```
---

## 🧬 Model Details

### 📊 Dataset Information
- **Classes**: 38 disease categories + healthy plants
- **Crops Covered**: Apple, Corn, Grape, Tomato, Potato, Pepper, Strawberry, etc.
- **Data Split**: 80% Train, 20% Validation

### 🎯 Supported Disease Categories
<details>
<summary>Click to expand full list (38 categories)</summary>

- Apple: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- Blueberry: Healthy
- Cherry: Powdery Mildew, Healthy
- Corn: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- Grape: Black Rot, Esca, Leaf Blight, Healthy
- Orange: Citrus Greening
- Peach: Bacterial Spot, Healthy
- Pepper: Bacterial Spot, Healthy
- Potato: Early Blight, Late Blight, Healthy
- Raspberry: Healthy
- Soybean: Healthy
- Squash: Powdery Mildew
- Strawberry: Leaf Scorch, Healthy
- Tomato: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy
</details>

### 🔍 Model Optimization
- **Model Size**: <50MB for efficient deployment
- **Inference Time**: <2 seconds per image
- **Memory Usage**: Optimized for mobile/edge deployment

---

## 🌾 Industry Applications

### 🚜 Smart Agriculture
- **Precision Farming**: Early disease detection for targeted treatment
- **Crop Monitoring**: Automated field surveillance systems
- **Yield Optimization**: Prevent crop losses through timely intervention
- **Resource Management**: Optimize pesticide and fertilizer usage

### 🔬 Research & Development
- **Plant Pathology Research**: Accelerate disease study workflows
- **Breeding Programs**: Screen disease-resistant varieties
- **Climate Impact Studies**: Monitor disease patterns under changing conditions
- **Agricultural Extension**: Support for field officers and consultants

### 📱 Commercial Applications
- **Mobile Apps**: Integration with farming applications
- **IoT Devices**: Embedded systems for continuous monitoring
- **Drone Surveillance**: Aerial crop health assessment
- **Robotic Systems**: Automated greenhouse management

### 🎓 Educational Tools
- **Agricultural Training**: Interactive learning platforms
- **University Research**: Academic project foundation
- **Extension Services**: Farmer education programs
- **Certification Programs**: Professional development resources

---

## 📊 Deployment Options

### ☁️ Cloud Deployment
- **Streamlit Cloud**: Current production deployment
- **AWS EC2**: Scalable cloud infrastructure
- **Google Cloud Platform**: AI/ML optimized environment
- **Azure ML**: Enterprise-grade deployment

### 📱 Edge Deployment
- **Raspberry Pi**: Farm-edge computing
- **NVIDIA Jetson**: High-performance edge AI
- **Mobile Apps**: React Native/Flutter integration
- **IoT Devices**: Embedded system deployment

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🎯 Priority Areas
- **Dataset Expansion**: Add more crop varieties and diseases
- **Model Optimization**: Improve accuracy and reduce inference time
- **Mobile App Development**: React Native/Flutter implementation
- **API Development**: RESTful API for enterprise integration
- **Documentation**: Improve user guides and tutorials

### 📝 Contribution Process
1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Make Changes**: Follow coding standards and add tests
4. **Commit Changes**: `git commit -m 'Add amazing feature'`
5. **Push to Branch**: `git push origin feature/amazing-feature`
6. **Open Pull Request**: Describe your changes and benefits

### 🧪 Development Guidelines
- **Code Quality**: Follow PEP 8 standards
- **Testing**: Add unit tests for new features
- **Documentation**: Update README and docstrings
- **Performance**: Ensure changes don't degrade model performance

---

## Output
[Output_sample_image](path)
## 📄 License & Citation

### 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 📚 Citation
If you use this project in your research, please cite:
```bibtex
@software{leafscan2024,
  title={LeafScan: Advanced Plant Disease Detection System},
  author={Alen K Aji},
  year={2025},
  url={[https://github.com/your-username/leafscan](https://github.com/AlenKAji/LeafScan)}
}
```

---

## 🙏 Acknowledgments

### 🎓 Research & Data
- **PlantVillage Dataset**: High-quality labeled images
- **Open Source Community**: Tools and frameworks

### 🛠️ Technology Partners
- **Streamlit**: Web application framework


---

## 📞 Contact & Support

### 👨‍💻 Author & Maintainer
**Alen K Aji**
- 📧 [Email](allenaji2512@gmail.com)
- 💼 [LinkedIn](https://www.linkedin.com/in/alenkaji/)
- 🐙 [GitHub](https://github.com/AlenKAji)

---

<div align="center">

### 🌱 Empowering Agriculture Through AI

**Built with ❤️ for sustainable farming and food security**

[⭐ Star this repository](https://github.com/your-username/leafscan) | [🚀 Try the App](https://leafscan-57gequp3qt3evqnenpnytv.streamlit.app/)

</div>
