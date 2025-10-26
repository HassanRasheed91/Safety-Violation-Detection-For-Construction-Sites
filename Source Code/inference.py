from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import uuid
import cv2
from pathlib import Path
import arabic_reshaper
from bidi.algorithm import get_display
from audio_feedback import AudioFeedback
import time
import torch

class SafetyViolationDetector:
    def __init__(self, model_path):
        """Initialize the YOLO model."""
        try:
            print(f"\n{'='*50}")
            print(f"Initializing Safety Violation Detector")
            print(f"{'='*50}")
            print(f"Model path: {model_path}")
            print(f"Current working directory: {os.getcwd()}")
            
            # Define violation classes to match data.yaml
            self.violation_classes = {
                'construction-machine': {
                    'en': 'Construction Machine',
                    'ar': 'آلة البناء',
                    'emoji': '🚜'
                },
                'fall_hazard': {
                    'en': 'Fall Hazard',
                    'ar': 'خطر السقوط',
                    'emoji': '⚠️'
                },
                'no-fall-protection': {
                    'en': 'No Fall Protection',
                    'ar': 'لا توجد حماية من السقوط',
                    'emoji': '⚠️'
                },
                'no-helmet': {
                    'en': 'No Helmet',
                    'ar': 'عدم ارتداء الخوذة',
                    'emoji': '⛑️'
                },
                'no-safety-glasses': {
                    'en': 'No Safety Glasses',
                    'ar': 'عدم ارتداء نظارات السلامة',
                    'emoji': '👓'
                },
                'no-safety-vest': {
                    'en': 'No Safety Vest',
                    'ar': 'عدم ارتداء سترة السلامة',
                    'emoji': '🦺'
                },
                'phone-usage': {
                    'en': 'Phone Usage',
                    'ar': 'استخدام الهاتف',
                    'emoji': '📱'
                },
                'using-mobile-beside-construction-machine': {
                    'en': 'Phone use near construction machine',
                    'ar': 'استخدام الهاتف قرب آلة البناء',
                    'emoji': '⚠️'
                },
                'poor-housekeeping': {
                    'en': 'Poor Housekeeping',
                    'ar': 'سوء التنظيم والترتيب',
                    'emoji': '🗑️'
                },
                'safe-ladder-usage': {
                    'en': 'Safe Ladder Usage',
                    'ar': 'استخدام آمن للسلم',
                    'emoji': '✅'
                },
                'unsafe_ladder_use': {
                    'en': 'Unsafe Ladder Usage',
                    'ar': 'استخدام غير آمن للسلم',
                    'emoji': '⚠️'
                },
                'unsecure- electricalwires': {
                    'en': 'Unsecure Electrical Wires',
                    'ar': 'أسلاك كهربائية غير آمنة',
                    'emoji': '⚡'
                }
            }
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            print("\nLoading YOLO model...")
            # Load model with device specification
            self.model = YOLO(model_path)
            
            # Force model to use CPU if CUDA is not available
            if not torch.cuda.is_available():
                print("CUDA not available, using CPU")
                self.model.to('cpu')
            else:
                print("Using CUDA for inference")
                self.model.to('cuda')
            
            # Verify model classes match our expected classes
            self._verify_model_classes()
            
            self.audio_feedback = AudioFeedback()
            
            # Try to load fonts that support both Arabic and emojis
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",  # Default Arial
                "C:/Windows/Fonts/arialbd.ttf",  # Arial Bold
                "C:/Windows/Fonts/seguiemj.ttf",  # Windows Segoe UI Emoji
                "C:/Windows/Fonts/calibri.ttf",   # Calibri
                "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS
                "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # Linux
                "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",  # Linux Arabic
            ]
            
            self.emoji_font_path = None
            self.text_font_path = None
            
            # Find emoji font
            for path in ["C:/Windows/Fonts/seguiemj.ttf", "/System/Library/Fonts/Apple Color Emoji.ttc"]:
                if os.path.exists(path):
                    self.emoji_font_path = path
                    break
            
            # Find text font (with Arabic support)
            for path in font_paths:
                if os.path.exists(path):
                    self.text_font_path = path
                    break
            
            if not self.emoji_font_path:
                print("Warning: No emoji font found. Emojis might not display correctly.")
                self.emoji_font_path = self.text_font_path
            
            if not self.text_font_path:
                print("Warning: No suitable font found. Using default font.")
                self.text_font_path = "arial.ttf"

        except Exception as e:
            print(f"Error in __init__ method: {e}")
            raise

    def _verify_model_classes(self):
        """Verify that model classes match our expected classes"""
        print("\nVerifying model classes...")
        model_classes = set(self.model.names.values())
        expected_classes = set(self.violation_classes.keys())
        
        print("\nModel Classes (from YOLO):")
        for idx, class_name in self.model.names.items():
            print(f"- {idx}: {class_name}")
        
        print("\nOur Expected Classes:")
        for class_name in self.violation_classes:
            print(f"- {class_name}")
        
        # Check for exact matches
        matching_classes = model_classes.intersection(expected_classes)
        missing_classes = expected_classes - model_classes
        extra_classes = model_classes - expected_classes
        
        print(f"\nClass Analysis:")
        print(f"✓ Matching classes: {len(matching_classes)}")
        print(f"✗ Missing classes: {len(missing_classes)}")
        print(f"⚠ Extra classes: {len(extra_classes)}")
        
        if matching_classes:
            print("\n✓ Matching classes:")
            for cls in sorted(matching_classes):
                print(f"  - {cls}")
        
        if missing_classes:
            print("\n✗ Missing classes (not in model):")
            for cls in sorted(missing_classes):
                print(f"  - {cls}")
        
        if extra_classes:
            print("\n⚠ Extra classes (in model but not expected):")
            for cls in sorted(extra_classes):
                print(f"  - {cls}")
        
        if not matching_classes:
            print("\n❌ CRITICAL: No classes match between model and expected classes!")
            print("This indicates a serious mismatch in the model training.")
        
        return len(matching_classes) > 0

    def _test_model(self):
        """Test if model is working correctly"""
        try:
            print("\nTesting model...")
            # Create a simple test image (black square)
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            test_path = "test_image.jpg"
            cv2.imwrite(test_path, test_img)
            
            # Try to run inference
            results = self.model(test_path, conf=0.5)[0]  #  confidence for testing
            
            print("\nModel Test Results:")
            print(f"Model type: {type(self.model)}")
            print(f"Number of classes: {len(self.model.names)}")
            print("\nAvailable classes:")
            for idx, class_name in self.model.names.items():
                print(f"- {idx}: {class_name}")
            
            # Clean up test image
            if os.path.exists(test_path):
                os.remove(test_path)
                
        except Exception as e:
            print(f"Model test failed: {e}")
            raise

    def _process_arabic_text(self, text):
        """Process Arabic text for proper rendering"""
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except Exception as e:
            print(f"Warning: Could not process Arabic text: {e}")
            return text

    def _process_image(self, image_path):
        """Process image to maintain high quality"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def _calculate_font_size(self, img_width, box_width):
        base_size = int(img_width * 0.02)
        box_adjusted = int(box_width * 0.15)
        return min(base_size, box_adjusted, 50)

    def _get_text_dimensions(self, text, font):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _adjust_font_size(self, draw, text_en, text_ar, box_width, img_width):
        """Adjust font size dynamically based on image and box size"""
        # Calculate base font size based on image width (more conservative)
        base_size = int(img_width * 0.03)  # Increased from 0.015 to 0.03
        box_adjusted = int(box_width * 0.18)  # Increased from 0.12 to 0.18
        
        # Use the smaller of the two, with a maximum limit
        initial_size = min(base_size, box_adjusted, 48)  # Increased max from 25 to 48
        font_size = initial_size
        
        while font_size > 16:  # Increased minimum from 8 to 16
            try:
                font = ImageFont.truetype(self.text_font_path, font_size)
                emoji_font = ImageFont.truetype(self.emoji_font_path, font_size)
                
                en_width, _ = self._get_text_dimensions(text_en, font)
                ar_width, _ = self._get_text_dimensions(text_ar, font)
                emoji_width, _ = self._get_text_dimensions("📱", emoji_font)
                
                # Calculate total width needed
                total_width = en_width + ar_width + emoji_width + (font_size * 1.5)
                
                # Check if text fits within box width
                if total_width <= box_width * 1.2:
                    return font, emoji_font, font_size
                
                font_size -= 1  # Smaller decrement for finer control
            except Exception as e:
                print(f"Font error: {e}")
                font_size -= 1
        
        # Fallback to minimum size
        font = ImageFont.truetype(self.text_font_path, 16)
        emoji_font = ImageFont.truetype(self.emoji_font_path, 16)
        return font, emoji_font, 16

    def _draw_box_with_text(self, draw, box, class_name, confidence, img):
        """Draw bounding box and text with proper formatting"""
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        img_width = img.size[0]
        
        # Get class info
        class_info = self.violation_classes.get(class_name, {
            'en': class_name,
            'ar': class_name,
            'emoji': '❓'
        })
        
        # Prepare text components
        text_en = f"{class_info['en']} ({confidence:.0%})"
        text_ar = self._process_arabic_text(class_info['ar'])
        emoji = class_info['emoji']
        
        # Draw box
        is_violation = not class_name.startswith('safe-') and class_name != 'site-mess'
        color = (255, 0, 0) if is_violation else (0, 255, 0)
        line_width = max(1, int(img_width/300))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # Get properly sized fonts
        font, emoji_font, font_size = self._adjust_font_size(draw, text_en, text_ar, box_width, img_width)
        padding = font_size // 4
        img_height = img.size[1]
        # Helper to keep label inside image
        def adjust_label_position(text_x, text_y, total_width, total_height):
            # Shift horizontally if needed
            if text_x < 0:
                text_x = 0
            elif text_x + total_width > img_width:
                text_x = img_width - total_width - padding
            # Shift vertically if needed
            if text_y < 0:
                text_y = y2 + 4  # Move below the box if above is out
                if text_y + total_height > img_height:
                    text_y = img_height - total_height - padding
            elif text_y + total_height > img_height:
                text_y = img_height - total_height - padding
            return text_x, text_y
        if class_name == 'using-mobile-beside-construction-machine':
            # Split English text into two lines for better fit
            base_en = class_info['en']
            if 'near' in base_en:
                en_parts = base_en.split('near')
                text_en = f"{en_parts[0].strip()} near\n{en_parts[1].strip()} ({confidence:.0%})"
            else:
                text_en = f"{base_en}\n({confidence:.0%})"
            # Split Arabic text into two lines for better fit
            base_ar = class_info['ar']
            if 'قرب' in base_ar:
                ar_parts = base_ar.split('قرب')
                text_ar = self._process_arabic_text(f"{ar_parts[0].strip()} قرب\n{ar_parts[1].strip()}")
            else:
                text_ar = self._process_arabic_text(base_ar)
            # Calculate text widths and heights for two lines
            emoji_bbox = draw.textbbox((0, 0), emoji, font=emoji_font)
            emoji_width = emoji_bbox[2] - emoji_bbox[0]
            emoji_height = emoji_bbox[3] - emoji_bbox[1]
            # Split text_en into lines
            en_lines = text_en.split('\n')
            en_line_bboxes = [draw.textbbox((0, 0), line, font=font) for line in en_lines]
            en_line_widths = [bbox[2] - bbox[0] for bbox in en_line_bboxes]
            en_line_heights = [bbox[3] - bbox[1] for bbox in en_line_bboxes]
            en_width = max(en_line_widths)
            en_height = sum(en_line_heights)
            ar_bbox = draw.textbbox((0, 0), text_ar, font=font)
            ar_width = ar_bbox[2] - ar_bbox[0]
            ar_height = ar_bbox[3] - ar_bbox[1]
            # The widest line
            total_width = emoji_width + padding + max(en_width, ar_width)
            total_height = en_height + ar_height + (padding * (len(en_lines) + 2))
            # Position above the box (default)
            text_y = y1 - total_height - 4
            text_x = x1
            # Adjust position to keep label inside image
            text_x, text_y = adjust_label_position(text_x, text_y, total_width + padding * 2, total_height + padding)
            # Draw background
            background_bbox = (
                text_x - padding,
                text_y - padding,
                text_x + total_width + padding * 2,
                text_y + total_height + padding
            )
            draw.rectangle(background_bbox, fill=color)
            # Draw emoji (vertically centered on all lines)
            emoji_y = text_y + padding
            draw.text((text_x, emoji_y), emoji, fill=(255, 255, 255), font=emoji_font)
            # Draw English (multiple lines) with background
            en_x = text_x + emoji_width + padding
            en_y = emoji_y
            for idx, line in enumerate(en_lines):
                line_bbox = draw.textbbox((en_x, en_y), line, font=font)
                draw.rectangle([
                    line_bbox[0] - padding // 2,
                    line_bbox[1] - padding // 4,
                    line_bbox[2] + padding // 2,
                    line_bbox[3] + padding // 4
                ], fill=color)
                draw.text((en_x, en_y), line, fill=(255, 255, 255), font=font)
                en_y += en_line_heights[idx] + padding // 2
            # Draw Arabic (always below English) with background
            ar_x = en_x
            ar_y = en_y + padding
            ar_lines = text_ar.split('\n')
            for idx, ar_line in enumerate(ar_lines):
                ar_line_bbox = draw.textbbox((ar_x, ar_y), ar_line, font=font)
                draw.rectangle([
                    ar_line_bbox[0] - padding // 2,
                    ar_line_bbox[1] - padding // 4,
                    ar_line_bbox[2] + padding // 2,
                    ar_line_bbox[3] + padding // 4
                ], fill=color)
                draw.text((ar_x, ar_y), ar_line, fill=(255, 255, 255), font=font)
                ar_y += (ar_line_bbox[3] - ar_line_bbox[1]) + padding // 2
        else:
            # Calculate text position
            text_y = y1 - font_size - 4
            current_x = x1
            # Adjust position to keep label inside image
            text_x, text_y = adjust_label_position(x1, text_y, box_width + padding * 2, font_size + padding)
            current_x = text_x
            # Calculate total text dimensions for background
            total_width = 0
            max_height = 0
            # Emoji dimensions
            emoji_bbox = draw.textbbox((0, 0), emoji, font=emoji_font)
            emoji_width = emoji_bbox[2] - emoji_bbox[0]
            emoji_height = emoji_bbox[3] - emoji_bbox[1]
            total_width += emoji_width + padding
            max_height = max(max_height, emoji_height)
            # English text dimensions
            en_bbox = draw.textbbox((0, 0), text_en, font=font)
            en_width = en_bbox[2] - en_bbox[0]
            en_height = en_bbox[3] - en_bbox[1]
            total_width += en_width + padding
            max_height = max(max_height, en_height)
            # Arabic text dimensions
            ar_bbox = draw.textbbox((0, 0), text_ar, font=font)
            ar_width = ar_bbox[2] - ar_bbox[0]
            ar_height = ar_bbox[3] - ar_bbox[1]
            total_width += ar_width
            max_height = max(max_height, ar_height)
            # Draw background
            background_bbox = (
                text_x - padding,
                text_y - padding,
                text_x + total_width + padding * 2,
                text_y + max_height + padding
            )
            draw.rectangle(background_bbox, fill=color)
            # Draw emoji
            draw.text((current_x, text_y), emoji, fill=(255, 255, 255), font=emoji_font)
            current_x += emoji_width + padding
            # Draw English text with background
            en_bbox = draw.textbbox((current_x, text_y), text_en, font=font)
            draw.rectangle([
                en_bbox[0] - padding // 2,
                en_bbox[1] - padding // 4,
                en_bbox[2] + padding // 2,
                en_bbox[3] + padding // 4
            ], fill=color)
            draw.text((current_x, text_y), text_en, fill=(255, 255, 255), font=font)
            # Draw Arabic text always below English with background
            en_height = en_bbox[3] - en_bbox[1]
            ar_y = text_y + en_height + padding // 2
            ar_bbox = draw.textbbox((current_x, ar_y), text_ar, font=font)
            draw.rectangle([
                ar_bbox[0] - padding // 2,
                ar_bbox[1] - padding // 4,
                ar_bbox[2] + padding // 2,
                ar_bbox[3] + padding // 4
            ], fill=color)
            draw.text((current_x, ar_y), text_ar, fill=(255, 255, 255), font=font)

    def _check_dangerous_phone_usage(self, detections):
        """Check if phone is being used dangerously near construction machines"""
        phone_detections = []
        machine_detections = []
        
        for det in detections:
            if det['class'] == 'phone-usage':
                phone_detections.append(det['box'])
            elif det['class'] == 'construction-machine':
                machine_detections.append(det['box'])
        
        # Check if any phone is near a machine
        for phone_box in phone_detections:
            for machine_box in machine_detections:
                # Calculate distance between phone and machine
                phone_center = ((phone_box[0] + phone_box[2])/2, (phone_box[1] + phone_box[3])/2)
                machine_center = ((machine_box[0] + machine_box[2])/2, (machine_box[1] + machine_box[3])/2)
                
                # Calculate distance
                distance = ((phone_center[0] - machine_center[0])**2 + (phone_center[1] - machine_center[1])**2)**0.5
                
                # If distance is less than 1/3 of the image width, consider it dangerous
                if distance < detections[0]['image_width'] * 0.33:
                    return True
        return False

    def _draw_danger_label(self, draw, img):
        """Draw a smaller red 'Person in danger' label at bottom center of image"""
        try:
            # Create the label text
            label_text = "PERSON IN DANGER!"
            label_ar = self._process_arabic_text("شخص في خطر!")
            
            # Calculate smaller dynamic font size based on image dimensions
            img_width, img_height = img.size
            base_font_size = min(img_width, img_height) // 35  # Reduced from 20 to 35 for smaller text
            font_size = max(min(base_font_size, 20), 10)  # Reduced max from 30 to 20, min from 15 to 10
            
            font = ImageFont.truetype(self.text_font_path, font_size)
            
            # Get text dimensions
            en_bbox = draw.textbbox((0, 0), label_text, font=font)
            ar_bbox = draw.textbbox((0, 0), label_ar, font=font)
            
            # Calculate total width and height
            total_width = max(en_bbox[2] - en_bbox[0], ar_bbox[2] - ar_bbox[0])
            total_height = (en_bbox[3] - en_bbox[1]) + (ar_bbox[3] - ar_bbox[1]) + 8  # Reduced spacing
            
            # Calculate position (bottom center of image)
            x = (img_width - total_width) // 2
            y = img_height - total_height - 15  # 15 pixels from bottom
            
            # Draw red background with smaller padding
            padding = font_size // 3  # Reduced padding
            background_bbox = [
                x - padding,
                y - padding,
                x + total_width + padding,
                y + total_height + padding
            ]
            
            # Draw red background
            draw.rectangle(background_bbox, fill=(255, 0, 0))
            
            # Draw white text
            draw.text((x, y), label_text, fill=(255, 255, 255), font=font)
            draw.text((x, y + (en_bbox[3] - en_bbox[1]) + 3), label_ar, fill=(255, 255, 255), font=font)  # Reduced spacing
            
            print("⚠️  Small red 'PERSON IN DANGER' label drawn at bottom center")
            
        except Exception as e:
            print(f"Warning: Could not draw danger label: {e}")

    def predict(self, image_path):
        """Run inference on an image with proper preprocessing and return detections"""
        try:
            print(f"\n{'='*60}")
            print(f"SAFETY VIOLATION DETECTION - STARTING ANALYSIS")
            print(f"{'='*60}")
            print(f"Input image: {image_path}")
            
            # Step 1: Validate image file
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Step 2: Load and preprocess image
            print("\nStep 1: Loading and preprocessing image...")
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            original_shape = img_cv.shape
            print(f"Original image shape: {original_shape}")
            
            # Convert BGR to RGB (YOLO expects RGB)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Step 3: Model information and verification
            print("\nStep 2: Model verification...")
            print(f"Model classes: {self.model.names}")
            print(f"Number of classes: {len(self.model.names)}")
            
            # Verify class mapping
            model_classes = list(self.model.names.values())
            expected_classes = list(self.violation_classes.keys())
            
            print("\nClass mapping verification:")
            for i, model_class in enumerate(model_classes):
                if model_class in expected_classes:
                    print(f"✓ {i}: {model_class} -> {self.violation_classes[model_class]['en']}")
                else:
                    print(f"✗ {i}: {model_class} -> NOT FOUND IN MAPPING")
            
            # Step 4: Run inference with proper preprocessing
            print("\nStep 3: Running YOLO inference...")
            
            # Use single confidence threshold of 0.25
            conf_thresh = 0.1
            print(f"Using confidence threshold: {conf_thresh}")
            results = self.model(img_rgb, conf=conf_thresh, verbose=False)[0]
            
            print(f"\nDetection Results:")
            print(f"Number of detections: {len(results.boxes)}")
            
            if len(results.boxes) == 0:
                print("No detections found with confidence threshold 0.25!")
                return {
                    "violations": [],
                    "has_violations": False,
                    "annotated_image": None,
                    "message": "No safety violations detected in the image."
                }
            
            print(f"Total detections: {len(results.boxes)}")
            
            # Step 5: Process detections
            print("\nStep 4: Processing detections...")
            
            # Convert back to PIL for drawing
            img_pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil)
            
            violations = []
            has_violations = False
            violation_messages_en = []
            violation_messages_ar = []
            all_detections = []
            
            # Track specific violations
            phone_boxes = []
            machine_boxes = []
            detections_to_draw = []
            print("\nDetailed detection analysis:")
            detected_classes_debug = []
            for i, (box, cls, conf) in enumerate(zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf)):
                class_id = int(cls)
                class_name = results.names[class_id]
                detected_classes_debug.append(class_name)
                box_coords = box.tolist()
                print(f"\nDetection #{i+1}:")
                print(f"  - Class ID: {class_id}")
                print(f"  - Class Name: '{class_name}'")
                print(f"  - Confidence: {conf:.3f}")
                print(f"  - Bounding Box: [{box_coords[0]:.1f}, {box_coords[1]:.1f}, {box_coords[2]:.1f}, {box_coords[3]:.1f}]")
                if class_name not in self.violation_classes:
                    print(f"  - WARNING: Class '{class_name}' not found in violation_classes!")
                    continue
                if class_name == 'phone-usage':
                    phone_boxes.append({'box': box_coords, 'confidence': conf})
                elif class_name == 'construction-machine':
                    machine_boxes.append({'box': box_coords, 'confidence': conf})
                else:
                    detections_to_draw.append({'class': class_name, 'box': box_coords, 'confidence': conf})
            print("All detected classes in this image:", detected_classes_debug)
            # Check for phone and construction machine anywhere in the image
            if phone_boxes and machine_boxes:
                # If both are present, only output the combined label for all phones
                for phone in phone_boxes:
                    self._draw_box_with_text(
                        draw,
                        phone['box'],
                        'using-mobile-beside-construction-machine',
                        phone['confidence'],
                        img_pil
                    )
                    has_violations = True
                    violations.append({
                        "class": "using-mobile-beside-construction-machine",
                        "label": {
                            "en": "Phone use near construction machine",
                            "ar": "استخدام الهاتف قرب آلة البناء"
                        },
                        "emoji": "⚠️",
                        "confidence": round(float(phone['confidence']), 3),
                        "bbox": {
                            "x1": round(phone['box'][0], 2),
                            "y1": round(phone['box'][1], 2),
                            "x2": round(phone['box'][2], 2),
                            "y2": round(phone['box'][3], 2)
                        }
                    })
                violation_messages_en.append("CRITICAL: Phone use near construction machine!")
                violation_messages_ar.append(self._process_arabic_text("حرج: استخدام الهاتف قرب آلة البناء!"))
            else:
                # Draw and report all other detections as usual
                for det in phone_boxes:
                    self._draw_box_with_text(draw, det['box'], 'phone-usage', det['confidence'], img_pil)
                    is_violation = True
                    violations.append({
                        "class": 'phone-usage',
                        "label": {
                            "en": self.violation_classes['phone-usage']['en'],
                            "ar": self.violation_classes['phone-usage']['ar']
                        },
                        "emoji": self.violation_classes['phone-usage']['emoji'],
                        "confidence": round(float(det['confidence']), 3),
                        "bbox": {
                            "x1": round(det['box'][0], 2),
                            "y1": round(det['box'][1], 2),
                            "x2": round(det['box'][2], 2),
                            "y2": round(det['box'][3], 2)
                        }
                    })
                    violation_messages_en.append(f"{self.violation_classes['phone-usage']['en']} ({det['confidence']:.1%})")
                    violation_messages_ar.append(self._process_arabic_text(self.violation_classes['phone-usage']['ar']))
                for det in machine_boxes:
                    self._draw_box_with_text(draw, det['box'], 'construction-machine', det['confidence'], img_pil)
                    is_violation = True
                    violations.append({
                        "class": 'construction-machine',
                        "label": {
                            "en": self.violation_classes['construction-machine']['en'],
                            "ar": self.violation_classes['construction-machine']['ar']
                        },
                        "emoji": self.violation_classes['construction-machine']['emoji'],
                        "confidence": round(float(det['confidence']), 3),
                        "bbox": {
                            "x1": round(det['box'][0], 2),
                            "y1": round(det['box'][1], 2),
                            "x2": round(det['box'][2], 2),
                            "y2": round(det['box'][3], 2)
                        }
                    })
                    violation_messages_en.append(f"{self.violation_classes['construction-machine']['en']} ({det['confidence']:.1%})")
                    violation_messages_ar.append(self._process_arabic_text(self.violation_classes['construction-machine']['ar']))
            # Draw and report all other detections as usual
            for det in detections_to_draw:
                self._draw_box_with_text(draw, det['box'], det['class'], det['confidence'], img_pil)
                is_violation = not det['class'].startswith('safe-')
                if is_violation:
                    has_violations = True
                    violations.append({
                        "class": det['class'],
                        "label": {
                            "en": self.violation_classes[det['class']]['en'],
                            "ar": self.violation_classes[det['class']]['ar']
                        },
                        "emoji": self.violation_classes[det['class']]['emoji'],
                        "confidence": round(float(det['confidence']), 3),
                        "bbox": {
                            "x1": round(det['box'][0], 2),
                            "y1": round(det['box'][1], 2),
                            "x2": round(det['box'][2], 2),
                            "y2": round(det['box'][3], 2)
                        }
                    })
                    violation_messages_en.append(f"{self.violation_classes[det['class']]['en']} ({det['confidence']:.1%})")
                    violation_messages_ar.append(self._process_arabic_text(self.violation_classes[det['class']]['ar']))
            
            # Step 6: Check for dangerous combinations
            print("\nStep 5: Checking for dangerous combinations...")
            
            if has_violations:
                summary = f"Found {len(violations)} safety violation(s)"
                text_en = "Safety violations detected: " + ", ".join(violation_messages_en)
                text_ar = "تم اكتشاف مخالفات السلامة: " + "، ".join(violation_messages_ar)
                print(f"✓ {summary}")
            else:
                summary = "No safety violations detected"
                text_en = "No safety violations detected - All practices appear safe"
                text_ar = "لا توجد مخالفات سلامة - جميع الممارسات تبدو آمنة"
                print(f"✓ {summary}")
                
                # Draw success message on image
                self._draw_success_message(draw, img_pil, text_en, text_ar)
            
            # Audio feedback
            self.audio_feedback.speak(text_en, text_ar)
            
            # Step 8: Save annotated image
            print("\nStep 7: Saving annotated image...")
            output_filename = f'output_{uuid.uuid4().hex[:8]}.jpg'
            output_path = os.path.join('static', output_filename)
            img_pil.save(output_path, 'JPEG', quality=95, optimize=True)
            print(f"✓ Annotated image saved: {output_filename}")
            
            # Step 9: Return results
            print(f"\n{'='*60}")
            print(f"ANALYSIS COMPLETE")
            print(f"{'='*60}")
            print(f"Total violations: {len(violations)}")
            print(f"Has violations: {has_violations}")
            print(f"Output image: {output_filename}")
            
            return {
                "violations": violations,
                "has_violations": has_violations,
                "annotated_image": output_filename,
                "message": summary,
                "confidence_threshold_used": conf_thresh,
                "total_detections": len(results.boxes)
            }
            
        except Exception as e:
            print(f"\n❌ ERROR during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "violations": [],
                "has_violations": False,
                "annotated_image": None,
                "error": str(e)
            }

    def _draw_success_message(self, draw, img, text_en, text_ar):
        """Draw a success message when no violations are detected"""
        try:
            font_size = self._calculate_font_size(img.size[0], img.size[0])
            font = ImageFont.truetype(self.text_font_path, font_size)
            emoji_font = ImageFont.truetype(self.emoji_font_path, font_size)
            
            # Calculate text dimensions
            emoji_bbox = draw.textbbox((0, 0), "✅", font=emoji_font)
            en_bbox = draw.textbbox((0, 0), text_en, font=font)
            ar_bbox = draw.textbbox((0, 0), text_ar, font=font)
            
            total_width = (emoji_bbox[2] - emoji_bbox[0]) + (en_bbox[2] - en_bbox[0]) + (ar_bbox[2] - ar_bbox[0]) + (font_size * 2)
            max_height = max(emoji_bbox[3] - emoji_bbox[1], en_bbox[3] - en_bbox[1], ar_bbox[3] - ar_bbox[1])
            
            # Center position
            x = (img.size[0] - total_width) // 2
            y = (img.size[1] - max_height) // 2
            
            # Draw background
            padding = font_size // 2
            background_bbox = (
                x - padding,
                y - padding,
                x + total_width + padding,
                y + max_height + padding
            )
            draw.rectangle(background_bbox, fill=(0, 255, 0))
            
            # Draw text components
            current_x = x
            draw.text((current_x, y), "✅", fill=(255, 255, 255), font=emoji_font)
            current_x += (emoji_bbox[2] - emoji_bbox[0]) + font_size
            
            draw.text((current_x, y), text_en, fill=(255, 255, 255), font=font)
            current_x += (en_bbox[2] - en_bbox[0]) + font_size
            
            draw.text((current_x, y), text_ar, fill=(255, 255, 255), font=font)
            
        except Exception as e:
            print(f"Warning: Could not draw success message: {e}")

# For testing
if __name__ == "__main__":
    detector = SafetyViolationDetector()
    if os.path.exists('image.jpg'):
        result = detector.predict('image.jpg')
        print(result) 