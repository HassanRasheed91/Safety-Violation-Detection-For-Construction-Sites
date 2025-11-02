# 🏗️ Safety Violation Detection for Construction Sites

> ⚠️ A real-time AI-powered web application that automatically detects PPE violations and safety hazards on construction sites using YOLOv8 deep learning with bilingual audio feedback.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green.svg)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-2.0+-black.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

Construction sites are high-risk environments where safety violations can lead to serious injuries or fatalities. This system leverages **YOLOv8** object detection to automatically identify safety violations in real-time, providing instant visual and audio feedback in both **English and Arabic**.

### 🎯 Objectives

- 🔍 **Real-time Detection** - Identify PPE violations and safety hazards instantly
- 🦺 **PPE Compliance** - Detect missing helmets, vests, gloves, and safety gear
- 🚧 **Hazard Recognition** - Identify unsafe proximity to equipment and restricted zones
- 🔊 **Bilingual Alerts** - Audio feedback in English and Arabic
- 🌐 **Web Interface** - Easy-to-use upload and visualization system
- 📊 **Visual Annotations** - Automatic labeling of detected violations

---

## ✨ Key Features

- 🤖 **YOLOv8 Detection** - State-of-the-art object detection model
- 🎨 **Visual Feedback** - Bounding boxes with violation labels
- 🔊 **Audio Alerts** - Bilingual voice notifications (English/Arabic)
- 🌐 **Web Application** - Flask-based interactive interface
- ⚡ **Real-time Processing** - Fast inference on uploaded images
- 🎯 **Adjustable Confidence** - Configurable detection thresholds
- 📦 **Custom Dataset** - Trained on construction site violations
- 💾 **Result Storage** - Automatic saving of annotated images

---

## 🦺 Detected Safety Violations

| Violation Type | Description | Severity |
|----------------|-------------|----------|
| ⛑️ **No Helmet** | Worker without hard hat | 🔴 Critical |
| 🦺 **No Safety Vest** | Missing high-visibility vest | 🔴 Critical |
| 🧤 **No Gloves** | Unprotected hands | 🟡 High |
| 👷 **No PPE** | Complete PPE absence | 🔴 Critical |
| 🚧 **Restricted Zone** | Entry into hazardous areas | 🔴 Critical |
| ⚠️ **Unsafe Proximity** | Too close to equipment/machinery | 🟠 Medium |
| 👔 **Improper Clothing** | Non-compliant work attire | 🟡 High |
| 🔧 **Equipment Misuse** | Incorrect tool/equipment usage | 🟠 Medium |

---

## 🛠️ Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| 🐍 **Python** | Core language | 3.8+ |
| 🎯 **YOLOv8** | Object detection | Latest |
| 🌐 **Flask** | Web framework | 2.0+ |
| 🔊 **gTTS** | Google Text-to-Speech | Latest |
| 📢 **pyttsx3** | Offline TTS engine | Latest |
| 🎮 **pygame** | Audio playback | Latest |
| 🖼️ **OpenCV** | Image processing | 4.5+ |
| 📊 **NumPy** | Numerical operations | 1.21+ |

---

## 🏗️ System Architecture

```
📸 Input Image
      ↓
🌐 Flask Web Interface
      ↓
🔍 YOLOv8 Model Inference
      ↓
📊 Detection Results
      ↓
   ├──────────────┬──────────────┐
   ↓              ↓              ↓
🎨 Visual      🔊 Audio      💾 Storage
   Annotation     Feedback       Saving
   ↓              ↓              ↓
📤 Annotated Image + Audio Alert
```

### 🧩 Core Components

#### 1️⃣ **Web Interface (Flask)**
- 📤 Image upload functionality
- 🖼️ Result visualization
- 📊 Detection statistics
- 💻 Responsive design

#### 2️⃣ **YOLOv8 Detection Engine**
- 🎯 Real-time object detection
- 🔍 10 violation categories
- ⚙️ Configurable confidence threshold
- ⚡ GPU-accelerated inference

#### 3️⃣ **Audio Feedback System**
- 🔊 English announcements (gTTS)
- 🔊 Arabic announcements (gTTS)
- 📢 Offline fallback (pyttsx3)
- 🎮 pygame audio playback

#### 4️⃣ **Annotation Module**
- 🎨 Dynamic label scaling
- 📏 Bounding box visualization
- 🌈 Color-coded severity levels
- 💾 Automatic result saving

---

## 💻 Installation

### 📋 Prerequisites

- ✅ Python 3.8 or higher
- ✅ pip package manager
- ✅ (Optional) CUDA-enabled GPU for faster inference

### 🚀 Setup Instructions

**1️⃣ Clone the repository**
```bash
git clone https://github.com/HassanRasheed91/Safety-Violation-Detection-For-Construction-Sites.git
cd Safety-Violation-Detection-For-Construction-Sites
```

**2️⃣ Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**3️⃣ Install dependencies**
```bash
pip install -r Important_libraries-to-install/requirements.txt
```

**4️⃣ Verify model weights**
Ensure `Model Weights/best.pt` exists in the directory.

### 📦 Required Libraries

```txt
ultralytics>=8.0.0
flask>=2.0.0
opencv-python>=4.5.0
numpy>=1.21.0
gtts>=2.3.0
pyttsx3>=2.90
pygame>=2.1.0
pillow>=9.0.0
```

---

## 🎮 Usage

### ▶️ Running the Application

```bash
python app.py
```

**Access the web interface:**
🌐 Open browser and navigate to `http://localhost:5000`

### 📸 Detection Workflow

1. **🌐 Open Web Interface** - Launch in browser
2. **📤 Upload Image** - Select construction site photo
3. **⏳ Processing** - YOLOv8 analyzes the image
4. **🔍 Detection** - Violations identified and annotated
5. **🔊 Audio Alert** - Bilingual announcement of violations
6. **📊 View Results** - Annotated image with bounding boxes
7. **💾 Save Results** - Automatic storage in `static/` folder

### 🎛️ Configuration

#### Adjusting Detection Sensitivity

Open `Source_Code/inference.py` and modify:

```python
conf_thresh = 0.25  # Adjust confidence threshold

# Threshold Guide:
# 0.05 - 0.15  : High sensitivity (more detections, more false positives)
# 0.20 - 0.30  : Balanced (recommended)
# 0.40 - 0.60  : Conservative (only high-confidence detections)
```

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 🟢 **0.10** | Very sensitive | Testing/Training |
| 🟡 **0.25** | Balanced | Production (Recommended) |
| 🔴 **0.50** | Conservative | High-precision scenarios |

---

## 📁 Project Structure

```
Safety-Violation-Detection/
│
├── 🌐 app.py                           # Flask application entry point
│
├── 📂 Source_Code/
│   ├── 🔍 inference.py                 # YOLOv8 detection logic
│   ├── 🔊 audio_feedback.py            # Bilingual audio system
│   ├── 📓 Model_training.ipynb         # Training notebook
│   ├── 📂 static/                      # Annotated outputs
│   └── 📂 uploads/                     # Uploaded images
│
├── 🤖 Model Weights/
│   └── 💾 best.pt                      # Trained YOLOv8 weights
│
├── 📊 Model's Training Results/        # Metrics and plots
│
├── 📦 Dataset/
│   └── My First Project.v3i.yolov8.zip
│
├── 📄 DATASET_INFO.txt                 # Dataset documentation
│
├── 📋 Important_libraries-to-install/
│   └── requirements.txt
│
├── 🎨 static/                          # Web assets
├── 📄 templates/
│   └── index.html                      # Frontend interface
│
└── 📤 uploads/                         # Temporary upload storage
```

---

## 🎓 Model Training

### 📊 Dataset

- **Format**: YOLOv8 annotated dataset
- **Classes**: 10 violation categories
- **Images**: Construction site photos
- **Annotations**: Bounding boxes with labels
- **Source**: Custom dataset + public construction safety images

### 🔄 Retraining the Model

**1️⃣ Prepare your dataset**
```bash
# Follow structure in DATASET_INFO.txt
Dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

**2️⃣ Open training notebook**
```bash
jupyter notebook Source_Code/Model_training.ipynb
```

**3️⃣ Train YOLOv8**
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train on custom dataset
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

**4️⃣ Replace model weights**
```bash
cp runs/detect/train/weights/best.pt "Model Weights/best.pt"
```

---

## 📈 Performance Metrics

### 🎯 Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| 🎯 **mAP@0.5** | 89.2% | Detection accuracy at 50% IoU |
| 📊 **Precision** | 87.5% | Correct detections ratio |
| 📈 **Recall** | 85.8% | Detection coverage |
| ⚡ **Inference Time** | <50ms | Per image on GPU |
| 📦 **Model Size** | 6.5MB | YOLOv8n variant |

### 📊 Per-Class Performance

| Violation | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| No Helmet | 91% | 88% | 89.5% |
| No Vest | 89% | 87% | 88.0% |
| No Gloves | 85% | 82% | 83.5% |

---

## 🔊 Audio Feedback System

### 🌍 Bilingual Support

#### 📢 English Announcements
```python
"Warning: Safety violation detected!"
"Worker without helmet identified"
"No safety vest detected"
```

#### 🔊 Arabic Announcements
```python
"تحذير: تم اكتشاف انتهاك للسلامة"
"تم تحديد عامل بدون خوذة"
"لم يتم الكشف عن سترة السلامة"
```

### 🎙️ Audio Engines

- **Primary**: gTTS (Google Text-to-Speech) - Online, high quality
- **Fallback**: pyttsx3 - Offline, system TTS
- **Playback**: pygame - Cross-platform audio

---

## 🚀 Future Enhancements

- 📹 **Video Stream Processing** - Real-time camera feed analysis
- 📱 **Mobile Application** - iOS/Android deployment
- ☁️ **Cloud Deployment** - Scalable web service
- 🤖 **Action Recognition** - Detect unsafe behaviors
- 📊 **Analytics Dashboard** - Violation statistics and trends
- 🔔 **SMS/Email Alerts** - Instant notifications
- 👥 **Multi-Person Tracking** - Individual worker monitoring
- 🎯 **Zone Detection** - Restricted area enforcement

---

## 🔧 Troubleshooting

### ❌ Common Issues

#### **📦 Module Not Found**
```bash
Solution:
pip install -r Important_libraries-to-install/requirements.txt
```

#### **🎯 Low Accuracy**
```bash
Solution:
- Use clear, well-lit images
- Adjust confidence threshold
- Retrain on more specific dataset
```

#### **🔊 No Audio Output**
```bash
Solution:
- Check system audio settings
- Install audio dependencies
- Try offline engine (pyttsx3)
```

#### **⚡ Slow Inference**
```bash
Solution:
- Install CUDA + cuDNN for GPU support
- Use smaller YOLOv8 variant (yolov8n)
- Reduce image resolution
```

---

## 🏭 Real-World Applications

### 🎯 Use Cases

- 🏗️ **Construction Sites** - Continuous safety monitoring
- 🏭 **Industrial Facilities** - Manufacturing floor compliance
- ⛏️ **Mining Operations** - PPE enforcement in hazardous zones
- 🔧 **Maintenance Work** - Safety gear verification
- 📹 **Security Systems** - Automated violation detection
- 📊 **Safety Audits** - Compliance documentation

### 💼 Benefits

- ✅ **Reduce Accidents** - Early hazard identification
- ✅ **Improve Compliance** - Automatic PPE verification
- ✅ **Cost Savings** - Prevent injuries and penalties
- ✅ **Documentation** - Automated violation logging
- ✅ **Worker Safety** - Proactive risk management

---

## 🤝 Contributing

Contributions are welcome! 🎉

### 📝 How to Contribute:

1. 🍴 Fork the repository
2. 🌿 Create feature branch (`git checkout -b feature/AmazingFeature`)
3. ✅ Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to branch (`git push origin feature/AmazingFeature`)
5. 🔃 Open a Pull Request

### 💡 Contribution Ideas:

- 🎥 Video processing support
- 📊 Advanced analytics dashboard
- 🌐 Multi-language support
- 🤖 Additional violation types
- 📱 Mobile app development

---

## 📄 License

This project is licensed under the MIT License. ⚖️

---

## 👨‍💻 Author

**Hassan Rasheed**

🎓 Machine Learning Engineer | Computer Vision Specialist

- 📧 **Email**: 221980038@gift.edu.pk
- 💼 **LinkedIn**: [hassan-rasheed-datascience](https://linkedin.com/in/hassan-rasheed-datascience)
- 🐙 **GitHub**: [HassanRasheed91](https://github.com/HassanRasheed91)

---

## 🙏 Acknowledgments

- 🤖 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - State-of-the-art object detection
- 🌐 [Flask](https://flask.palletsprojects.com/) - Web framework
- 🔊 [Google Text-to-Speech (gTTS)](https://github.com/pndurette/gTTS) - Audio feedback
- 🎮 [pygame](https://www.pygame.org/) - Audio playback
- 🏷️ [Roboflow](https://roboflow.com/) - Dataset annotation and management
- 🏗️ Construction safety research community

---

<div align="center">

### ⭐ Star this repo if you find it helpful!

**Made with ❤️ By Hassan Rasheed**

🔗 [View Project](https://github.com/HassanRasheed91/Safety-Violation-Detection-For-Construction-Sites) • 🐛 [Report Bug](https://github.com/HassanRasheed91/Safety-Violation-Detection-For-Construction-Sites/issues) • 💡 [Request Feature](https://github.com/HassanRasheed91/Safety-Violation-Detection-For-Construction-Sites/issues)

---

</div>
