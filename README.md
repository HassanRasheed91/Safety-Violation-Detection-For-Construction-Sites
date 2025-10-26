
# 🏗️ AI-Powered Safety Violation Detection System for Construction Sites

A real-time deep learning-based web application that detects safety violations on construction sites using YOLOv8, with visual and bilingual (English & Arabic) audio feedback.

---

## 🚀 Project Overview

This system is designed to enhance construction site safety by identifying violations using a YOLO-based deep learning model. It provides:

- **Real-time detection** of safety violations from uploaded images.
- **Audio feedback** in both English and Arabic.
- **Web interface** for easy interaction and result visualization.

---

## 📁 Folder Structure

```
.
├── app.py                             # Flask app entry point
├── Source_Code/
│   ├── inference.py                   # Core YOLO inference logic
│   ├── audio_feedback.py             # English & Arabic audio feedback
│   ├── Model_training.ipynb          # YOLO model training notebook
│   ├── static/                        # Output images (annotated)
│   └── uploads/                       # Uploaded images
├── Model Weights/
│   └── best.pt                        # Trained YOLO weights
├── Model's Training Results/         # Training metrics & plots
├── Dataset/
│   └── My First Project.v3i.yolov8.zip  # Dataset for training
├── DATASET_INFO.txt                  # Dataset structure & preparation guide
├── Important_libraries-to-install/
│   └── requirements.txt              # All required Python libraries
├── static/                           # Static assets for Flask app
├── templates/
│   └── index.html                    # Frontend HTML template
├── uploads/                          # Uploaded images
```

---

## ⚙️ How to Run This System

### Step 1: Install Python
Ensure Python 3.8+ is installed.  
🔗 [Download Python](https://www.python.org/downloads/)

### Step 2: Set Up a Virtual Environment

<details>
<summary>Windows</summary>

```bash
python -m venv venv
venv\Scripts\activate
```
</details>

<details>
<summary>Linux / Mac</summary>

```bash
python3 -m venv venv
source venv/bin/activate
```
</details>

### Step 3: Install Dependencies

```bash
pip install -r Important_libraries-to-install/requirements.txt
```

### Step 4: Check Model Weights

Ensure the file `Model Weights/best.pt` exists in the directory.

### Step 5: Run the Flask App

```bash
python app.py
```

Access the web interface at: [http://localhost:5000](http://localhost:5000)

---

## 🎯 Customizing Detection Threshold

To change the detection confidence threshold:

Open: `Source_Code/inference.py`

Find and modify:

```python
conf_thresh = 0.1  # Adjust as needed
results = self.model(img_rgb, conf=conf_thresh, verbose=False)[0]
```

| Threshold | Behavior |
|-----------|----------|
| `0.05`    | More sensitive, more false positives |
| `0.25`    | Balanced detection |
| `0.5`     | Only high-confidence violations shown |

---

## 🏷️ Dynamic Label Sizing

Violation labels are dynamically scaled based on bounding box size to ensure:

- Proper visibility within the image
- Fit for both English and Arabic labels

⚠️ Oversized labels may overflow or be clipped.

---

## 🧠 How to Use

1. Open the web interface in your browser.
2. Upload a construction site image.
3. The system detects and displays violations.
4. Bilingual audio feedback will describe the violations.
5. Annotated results will be displayed and stored.

---

## 🛠️ Dataset & Training

- Full dataset structure is described in `DATASET_INFO.txt`.
- Retrain the YOLO model using: `Source_Code/Model_training.ipynb`
- Replace `best.pt` after training for updated inference.

---

## 🧰 Troubleshooting Tips

- **Missing modules?** Run the pip install command again in your virtual environment.
- **Accuracy issues?** Use clear images with visible safety gear and common violations.
- **Want to retrain?** Modify dataset and follow instructions in the training notebook.
- **GPU support?** Make sure correct CUDA + PyTorch version is installed.

---

## 📚 References & Libraries Used

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [Flask Web Framework](https://flask.palletsprojects.com/)
- Audio: `gTTS`, `pyttsx3`, `pygame`
- Annotation Tools: [LabelImg](https://github.com/tzutalin/labelImg), [Roboflow](https://roboflow.com/)

---

## 🙌 Contribution & Support

Feel free to fork, raise issues, or contribute improvements via pull requests.

For questions or feedback, reach out via [GitHub Issues](https://github.com/YourRepo/issues).
