# Construction Site Violation Detector System

## Project Overview
This system detects safety violations on construction sites using a deep learning model (YOLO) and provides both visual and audio feedback in English and Arabic. It features a web interface for uploading images and viewing annotated results.

## Folder Structure & Purpose

- **app.py**
  - Main entry point. Runs the Flask web server and connects all components.

- **Source_Code/**
  - `inference.py`: Core logic for running YOLO inference, drawing labels, and handling all detection logic.
  - `audio_feedback.py`: Handles English and Arabic audio feedback for detected violations.
  - `Model_training.ipynb`: Jupyter notebook for training the YOLO model.
  - `static/`, `uploads/`: Used internally for storing outputs and uploads during inference.

- **Model Weights/**
  - `best.pt`: Trained YOLO model weights. Required for inference.

- **Model's Training Results/**
  - Contains training metrics, plots (confusion matrix, F1/PR/R curves), and results from model training.

- **Dataset/**
  - Contains the dataset used for training (e.g., `My First Project.v3i.yolov8.zip`).

- **DATASET_INFO.txt**
  - Documentation on dataset structure, classes, and how to prepare or use a dataset for training or retraining.

- **Important_libraries-to-install/**
  - `requirements.txt`: All required Python libraries for running and training the system.

- **static/**
  - Stores static files and output images for the web interface (annotated images, etc.).

- **templates/**
  - HTML templates for the web interface (e.g., `index.html`).

- **uploads/**
  - Temporary storage for images uploaded via the web interface.

---

## How to Run This System on a New Machine (Windows, Linux, Mac)

### 1. Install Python
- Ensure Python 3.8 or newer is installed.
- [Download Python](https://www.python.org/downloads/)

### 2. Create a Virtual Environment
**Windows:**
```sh
python -m venv venv
venv\Scripts\activate
```
**Linux/Mac:**
```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Libraries
```sh
pip install -r Important_libraries-to-install/requirements.txt
```

### 4. Model Weights
- Ensure `Model Weights/best.pt` exists. This file is required for the model to run.

### 5. Run the Application
```sh
python app.py
```
- The web interface will be available at [http://localhost:5000](http://localhost:5000)

---

## How to Change the Detection Threshold

- The detection threshold controls how confident the model must be before it reports a violation.
- To change it, open `Source_Code/inference.py` and look for the line in the `predict` method:
  ```python
  conf_thresh = 0.1  # Default value
  results = self.model(img_rgb, conf=conf_thresh, verbose=False)[0]
  ```
- Change `conf_thresh` to your desired value (e.g., 0.25 for stricter detection, 0.05 for more sensitive detection).

### What Happens with Different Thresholds?
- **Higher threshold (e.g., 0.5):**
  - Only very confident detections are reported.
  - Fewer false positives, but you may miss some real violations.
- **Lower threshold (e.g., 0.05):**
  - More detections, including less confident ones.
  - May catch more violations, but also more false positives (incorrect detections).

---

## Why Label Size is Dynamic

- The size of the violation label is dynamically calculated based on the image and bounding box size.
- **If you increase the label size too much, the full text may not fit inside the image, and parts of the label could be cut off or go outside the image.**
- The system uses a dynamic, optimal size to ensure the entire label (English and Arabic) is always visible and fits within the image boundaries, regardless of image or box size.

---

## Usage
1. Open the web interface in your browser.
2. Upload an image of a construction site.
3. The system will analyze the image, display detected violations, and provide audio feedback in English and Arabic.
4. Annotated images and results will be shown on the page.

---

## Dataset Preparation & Training
- See `DATASET_INFO.txt` for details on dataset structure, class names, and how to add or use a new dataset.
- To retrain the model, use the provided Jupyter notebook in `Source_Code/Model_training.ipynb` and follow the dataset structure described in `DATASET_INFO.txt`.

---

## Troubleshooting & Tips
- If you encounter missing module errors, ensure all libraries in `requirements.txt` are installed in your virtual environment.
- For best results, use clear images with visible safety equipment and violations.
- The system supports both English and Arabic labels and audio.
- If you change the dataset or retrain the model, update `Model Weights/best.pt` accordingly.
- For GPU acceleration, ensure you have the correct version of PyTorch and CUDA installed.

---

## Credits & References
- YOLO by Ultralytics: https://github.com/ultralytics/ultralytics
- Flask: https://flask.palletsprojects.com/
- gTTS, pyttsx3, pygame for audio feedback
- For dataset annotation: [LabelImg](https://github.com/tzutalin/labelImg), [Roboflow](https://roboflow.com/)

---

For further help, see the YOLO documentation or contact the project maintainer. 