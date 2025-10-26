from flask import Flask, request, jsonify, render_template, send_file
import os
from werkzeug.utils import secure_filename
import sys

# Add Source_Code directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
source_code_dir = os.path.join(current_dir, 'Source_Code')
if source_code_dir not in sys.path:
    sys.path.append(source_code_dir)

from inference import SafetyViolationDetector
import uuid

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = os.path.join('uploads')
STATIC_FOLDER = os.path.join('static')
MODEL_FOLDER = os.path.join('Model Weights')
TEMPLATES_FOLDER = os.path.join('templates')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.template_folder = TEMPLATES_FOLDER

# Initialize the detector with correct model path
model_path = os.path.join(MODEL_FOLDER, 'best.pt')
print(f"\nInitializing Safety Violation Detector")
print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

try:
    detector = SafetyViolationDetector(model_path)
    print("Detector initialized successfully!")
except Exception as e:
    print(f"Error initializing detector: {e}")
    raise

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename):
    """Generate a unique filename while preserving the original extension"""
    ext = os.path.splitext(original_filename)[1]
    return f"{uuid.uuid4().hex}{ext}"

@app.route('/')
def index():
    """Serve the upload form"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Endpoint to analyze an image for safety violations.
    Expects an image file in the request.
    """
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename
            filename = generate_unique_filename(secure_filename(file.filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save uploaded file
            file.save(filepath)
            
            # Run inference
            result = detector.predict(filepath)
            
            # Copy original image to static folder for display
            original_static_path = os.path.join(app.config['STATIC_FOLDER'], filename)
            with open(filepath, 'rb') as f:
                with open(original_static_path, 'wb') as f2:
                    f2.write(f.read())
            
            # Add the original image path to the response
            result['original_image'] = f'/static/{filename}'
            result['annotated_image'] = f'/static/{result["annotated_image"]}'
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    print(f"Model folder exists: {os.path.exists(MODEL_FOLDER)}")
    print(f"Templates folder exists: {os.path.exists(TEMPLATES_FOLDER)}")
    app.run(debug=True, host='0.0.0.0', port=5000) 