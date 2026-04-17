"""
================================================================================
FULL-BODY DETECTION MODULE (Task 2.1)
================================================================================
Purpose: Detect if an image contains a COMPLETE human body (no cropped limbs)

WHAT IS YOLOv8?
===============
YOLO = "You Only Look Once" - a family of real-time object detection models.

History:
- YOLO v1 (2016): Revolutionary - first real-time object detector
- YOLO v2-v4: Incremental improvements
- YOLO v5: Made by Ultralytics, became very popular
- YOLO v8 (2023): Latest from Ultralytics, state-of-the-art

Why YOLOv8?
- FAST: Can process 100+ images per second
- ACCURATE: Competitive with much slower models
- EASY: Simple Python API via `ultralytics` package
- VERSATILE: Detection, Segmentation, Classification, AND Pose Estimation

YOLOv8-Pose specifically:
- Detects people AND their body keypoints (joints)
- Returns 17 keypoints per person (COCO format)
- Perfect for checking if full body is visible!

THE 17 COCO KEYPOINTS:
======================
Index | Body Part          | What to check
------|--------------------|--------------
  0   | Nose               | Face visible
  1   | Left Eye           | Face visible
  2   | Right Eye          | Face visible
  3   | Left Ear           | Face visible  
  4   | Right Ear          | Face visible
  5   | Left Shoulder      | Upper body
  6   | Right Shoulder     | Upper body
  7   | Left Elbow         | Arms
  8   | Right Elbow        | Arms
  9   | Left Wrist         | Hands (important!)
  10  | Right Wrist        | Hands (important!)
  11  | Left Hip           | Lower body
  12  | Right Hip          | Lower body
  13  | Left Knee          | Legs
  14  | Right Knee         | Legs
  15  | Left Ankle         | Feet (important!)
  16  | Right Ankle        | Feet (important!)

For FULL BODY, we especially need:
- At least one wrist visible (hands not cut off)
- At least one ankle visible (feet not cut off)
- Core body keypoints (shoulders, hips) visible

================================================================================
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from ultralytics import YOLO
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to your dataset - UPDATE THIS
DATASET_PATH = r"C:\Users\alank\OneDrive\Documents\image_filtering\image_filtering_pipeline\person-20260414T072836Z-3-001"

# Output directory for results
OUTPUT_DIR = "fullbody_results"

# YOLOv8 Pose model - will auto-download on first run
# Options: yolov8n-pose (nano/fast), yolov8s-pose (small), yolov8m-pose (medium), yolov8l-pose (large)
MODEL_NAME = "yolov8m-pose.pt"  # Medium model - good balance of speed and accuracy

# Confidence thresholds
PERSON_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to consider a person detection
KEYPOINT_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for a keypoint to be "visible"

# Full body criteria (see explanation below)
MIN_KEYPOINTS_VISIBLE = 13  # Out of 17 total keypoints


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FullBodyResult:
    """
    Stores the result of full-body analysis for one image.
    
    Using a dataclass makes the code cleaner and self-documenting.
    """
    image_path: str
    is_full_body: bool
    confidence_score: float  # 0-1, how confident we are
    num_people_detected: int
    keypoints_visible: int  # How many of 17 keypoints detected
    missing_keypoints: List[str]  # Which body parts are missing
    has_visible_feet: bool
    has_visible_hands: bool
    has_visible_face: bool
    error: Optional[str] = None


# Keypoint names for readable output
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Critical keypoints that MUST be visible for full body
# We group them to allow flexibility (e.g., one hand hidden is OK)
CRITICAL_KEYPOINT_GROUPS = {
    'face': [0, 1, 2, 3, 4],  # At least one face keypoint
    'hands': [9, 10],  # At least one wrist
    'feet': [15, 16],  # At least one ankle
    'core': [5, 6, 11, 12],  # Shoulders and hips
}


# =============================================================================
# FULL BODY DETECTOR CLASS
# =============================================================================

class FullBodyDetector:
    """
    Detects whether an image contains a complete (uncropped) human body.
    
    Uses YOLOv8-Pose to detect body keypoints and checks if critical
    body parts (hands, feet, face) are visible.
    
    Usage:
        detector = FullBodyDetector()
        result = detector.analyze_image("path/to/image.jpg")
        print(result.is_full_body)  # True or False
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the detector by loading the YOLOv8-Pose model.
        
        The model will be automatically downloaded on first run (~50MB).
        Subsequent runs use the cached model.
        """
        print(f"🔄 Loading YOLOv8-Pose model: {model_name}")
        print("   (First run will download the model, ~50MB)")
        
        # Load the model - ultralytics handles downloading
        self.model = YOLO(model_name)
        
        print("✅ Model loaded successfully!")
        
        # Store thresholds
        self.person_conf = PERSON_CONFIDENCE_THRESHOLD
        self.keypoint_conf = KEYPOINT_CONFIDENCE_THRESHOLD
    
    def analyze_image(self, image_path: str) -> FullBodyResult:
        """
        Analyze a single image for full-body presence.
        
        This is the main method you'll call for each image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            FullBodyResult with all analysis details
        """
        # Read image
        try:
            image = cv2.imread(image_path)
            if image is None:
                return FullBodyResult(
                    image_path=image_path,
                    is_full_body=False,
                    confidence_score=0.0,
                    num_people_detected=0,
                    keypoints_visible=0,
                    missing_keypoints=KEYPOINT_NAMES.copy(),
                    has_visible_feet=False,
                    has_visible_hands=False,
                    has_visible_face=False,
                    error="Failed to read image"
                )
        except Exception as e:
            return FullBodyResult(
                image_path=image_path,
                is_full_body=False,
                confidence_score=0.0,
                num_people_detected=0,
                keypoints_visible=0,
                missing_keypoints=KEYPOINT_NAMES.copy(),
                has_visible_feet=False,
                has_visible_hands=False,
                has_visible_face=False,
                error=str(e)
            )
        
        # Run YOLOv8-Pose inference
        # verbose=False suppresses per-image output
        results = self.model(image, verbose=False, conf=self.person_conf)
        
        # Process results
        return self._process_results(image_path, results, image.shape)
    
    def _process_results(self, image_path: str, results, image_shape: tuple) -> FullBodyResult:
        """
        Process YOLO results and determine if full body is present.
        
        Logic:
        1. If no person detected → not full body
        2. If multiple people → analyze the main/largest person
        3. Check which keypoints are visible with sufficient confidence
        4. Apply our full-body criteria
        """
        # Get the first result (we only passed one image)
        result = results[0]
        
        # Check if any people were detected
        if result.keypoints is None or len(result.keypoints) == 0:
            return FullBodyResult(
                image_path=image_path,
                is_full_body=False,
                confidence_score=0.0,
                num_people_detected=0,
                keypoints_visible=0,
                missing_keypoints=KEYPOINT_NAMES.copy(),
                has_visible_feet=False,
                has_visible_hands=False,
                has_visible_face=False,
                error=None
            )
        
        num_people = len(result.keypoints)
        
        # If multiple people, find the "main" person (largest bounding box)
        # This assumes the subject of the photo is the most prominent person
        if num_people > 1:
            main_person_idx = self._find_main_person(result)
        else:
            main_person_idx = 0
        
        # Get keypoints for the main person
        # Shape: (17, 3) where each row is [x, y, confidence]
        keypoints = result.keypoints[main_person_idx].data.cpu().numpy()[0]
        
        # Analyze keypoint visibility
        visible_keypoints = []
        missing_keypoints = []
        
        for idx, (x, y, conf) in enumerate(keypoints):
            if conf >= self.keypoint_conf:
                visible_keypoints.append(idx)
            else:
                missing_keypoints.append(KEYPOINT_NAMES[idx])
        
        # Check critical body part groups
        has_face = any(idx in visible_keypoints for idx in CRITICAL_KEYPOINT_GROUPS['face'])
        has_hands = any(idx in visible_keypoints for idx in CRITICAL_KEYPOINT_GROUPS['hands'])
        has_feet = any(idx in visible_keypoints for idx in CRITICAL_KEYPOINT_GROUPS['feet'])
        has_core = all(idx in visible_keypoints for idx in CRITICAL_KEYPOINT_GROUPS['core'])
        
        # Calculate visibility score
        num_visible = len(visible_keypoints)
        visibility_score = num_visible / 17.0
        
        # Determine if full body
        # Criteria:
        # 1. At least MIN_KEYPOINTS_VISIBLE keypoints detected
        # 2. At least one hand (wrist) visible
        # 3. At least one foot (ankle) visible
        # 4. Core body (shoulders + hips) visible
        
        is_full_body = (
            num_visible >= MIN_KEYPOINTS_VISIBLE and
            has_hands and
            has_feet and
            has_core
        )
        
        # Calculate confidence score (weighted combination)
        confidence_score = (
            0.4 * visibility_score +  # Overall visibility
            0.2 * float(has_hands) +   # Hands bonus
            0.2 * float(has_feet) +    # Feet bonus
            0.2 * float(has_core)      # Core bonus
        )
        
        return FullBodyResult(
            image_path=image_path,
            is_full_body=is_full_body,
            confidence_score=round(confidence_score, 3),
            num_people_detected=num_people,
            keypoints_visible=num_visible,
            missing_keypoints=missing_keypoints,
            has_visible_feet=has_feet,
            has_visible_hands=has_hands,
            has_visible_face=has_face,
            error=None
        )
    
    def _find_main_person(self, result) -> int:
        """
        Find the index of the "main" person in the image.
        
        Strategy: Return the person with the largest bounding box area.
        This assumes the main subject is most prominent in the frame.
        
        Alternative strategies (not implemented):
        - Person closest to center of image
        - Person with highest detection confidence
        - Person with most visible keypoints
        """
        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) format: x1, y1, x2, y2
        
        max_area = 0
        main_idx = 0
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                main_idx = idx
        
        return main_idx
    
    def visualize_result(self, image_path: str, result: FullBodyResult, output_path: str = None):
        """
        Create a visualization showing detected keypoints and result.
        
        This is useful for debugging and understanding why images pass/fail.
        
        Args:
            image_path: Path to original image
            result: The FullBodyResult from analyze_image()
            output_path: Where to save visualization (optional)
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        
        # Run model again to get keypoints for visualization
        results = self.model(image, verbose=False, conf=self.person_conf)
        
        # Draw keypoints using YOLOv8's built-in plotting
        annotated = results[0].plot()
        
        # Add our analysis result as text overlay
        status = "✓ FULL BODY" if result.is_full_body else "✗ NOT FULL BODY"
        color = (0, 255, 0) if result.is_full_body else (0, 0, 255)
        
        # Add text with background for readability
        text = f"{status} (conf: {result.confidence_score:.2f})"
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, color, 2)
        
        details = f"Keypoints: {result.keypoints_visible}/17 | Hands: {result.has_visible_hands} | Feet: {result.has_visible_feet}"
        cv2.putText(annotated, details, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        
        if output_path:
            cv2.imwrite(output_path, annotated)
        
        return annotated


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_dataset(dataset_path: str, output_dir: str, max_images: int = None) -> List[FullBodyResult]:
    """
    Process all images in a dataset and save results.
    
    Args:
        dataset_path: Root folder containing images
        output_dir: Where to save results and visualizations
        max_images: Optional limit for testing (None = process all)
        
    Returns:
        List of FullBodyResult for all processed images
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"\n📊 Processing {len(image_paths)} images...")
    
    # Initialize detector
    detector = FullBodyDetector()
    
    # Process images
    results = []
    full_body_count = 0
    
    for i, img_path in enumerate(image_paths):
        # Progress update
        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f"   Processed {i + 1}/{len(image_paths)} images "
                  f"({100*(i+1)/len(image_paths):.1f}%)")
        
        # Analyze image
        result = detector.analyze_image(img_path)
        results.append(result)
        
        if result.is_full_body:
            full_body_count += 1
        
        # Save visualization for first 20 images (for debugging)
        if i < 20:
            vis_path = os.path.join(output_dir, "visualizations", 
                                   f"{i:03d}_{Path(img_path).stem}.jpg")
            detector.visualize_result(img_path, result, vis_path)
    
    # Print summary
    print(f"\n📈 RESULTS SUMMARY:")
    print(f"   Total images: {len(results)}")
    print(f"   Full body detected: {full_body_count} ({100*full_body_count/len(results):.1f}%)")
    print(f"   Not full body: {len(results) - full_body_count} ({100*(len(results)-full_body_count)/len(results):.1f}%)")
    
    # Save detailed results to JSON
    results_data = []
    for r in results:
        results_data.append({
            'image_path': r.image_path,
            'filename': os.path.basename(r.image_path),
            'is_full_body': r.is_full_body,
            'confidence_score': r.confidence_score,
            'num_people_detected': r.num_people_detected,
            'keypoints_visible': r.keypoints_visible,
            'missing_keypoints': r.missing_keypoints,
            'has_visible_feet': r.has_visible_feet,
            'has_visible_hands': r.has_visible_hands,
            'has_visible_face': r.has_visible_face,
            'error': r.error
        })
    
    results_path = os.path.join(output_dir, "fullbody_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    print(f"🖼️  Visualizations saved to: {output_dir}/visualizations/")
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🏃 FULL-BODY DETECTION MODULE")
    print("="*70)
    print("""
This module uses YOLOv8-Pose to detect if images contain full human bodies.

WHAT IT CHECKS:
✓ All major body keypoints visible (head, shoulders, hips, knees, ankles)
✓ At least one hand (wrist) visible
✓ At least one foot (ankle) visible
✓ Core body structure intact

WHAT IT REJECTS:
✗ Cropped images (missing feet, hands, etc.)
✗ Partial body shots (torso only, legs only)
✗ Images where body is hidden behind objects
    """)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at: {DATASET_PATH}")
        print("   Please update DATASET_PATH in this script.")
        exit(1)
    
    # Process dataset
    # Start with a small test (max_images=50) to verify everything works
    print("\n🧪 Running test on first 50 images...")
    print("   (Remove max_images parameter to process full dataset)\n")
    
    results = process_dataset(
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        max_images=50  # Remove this line to process all images
    )
    
    print("\n" + "="*70)
    print("✅ FULL-BODY DETECTION COMPLETE!")
    print("="*70)
    print("""
NEXT STEPS:
1. Check the 'visualizations/' folder to verify detections look correct
2. Adjust thresholds if needed:
   - KEYPOINT_CONFIDENCE_THRESHOLD (lower = more lenient)
   - MIN_KEYPOINTS_VISIBLE (lower = accept more partial bodies)
3. Integrate this module into your main pipeline
    """)
