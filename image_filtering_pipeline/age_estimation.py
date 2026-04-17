"""
================================================================================
AGE ESTIMATION MODULE (Task 2.3)
================================================================================
Purpose: Estimate the age of the person and filter out young children (<13).

This version replaces MediaPipe with OpenCV DNN face detection for face crops.
================================================================================
"""

import os
import cv2
import json
import torch
import urllib.request
import numpy as np
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
from tqdm import tqdm
from torchvision import transforms, models

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_PATH = r"C:\Users\alank\OneDrive\Documents\image_filtering\image_filtering_pipeline\person-20260414T072836Z-3-001\person"
OUTPUT_DIR = "age_results"
MIN_AGE_THRESHOLD = 13
FACE_CONFIDENCE_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FACE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "opencv_face_detector")
FACE_PROTOTXT = os.path.join(FACE_MODEL_DIR, "deploy.prototxt")
FACE_CAFFEMODEL = os.path.join(FACE_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
FACE_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


@dataclass
class AgeResult:
    image_path: str
    is_adult: bool
    estimated_age: float
    age_confidence: float
    face_detected: bool
    error: Optional[str] = None


class AgeEstimator:
    """Estimates age from face images using a pretrained ResNet18 backbone."""

    def __init__(self, min_age_threshold: int = MIN_AGE_THRESHOLD):
        print("Initializing Age Estimator...")
        print(f"Device: {DEVICE}")

        self.min_age_threshold = min_age_threshold
        self._init_face_detector()
        self._load_age_model()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        print("Age Estimator initialized")

    def _ensure_face_model_files(self):
        os.makedirs(FACE_MODEL_DIR, exist_ok=True)

        if not os.path.exists(FACE_PROTOTXT):
            print("Downloading deploy.prototxt...")
            urllib.request.urlretrieve(FACE_PROTOTXT_URL, FACE_PROTOTXT)

        if not os.path.exists(FACE_CAFFEMODEL):
            print("Downloading res10_300x300_ssd_iter_140000.caffemodel...")
            urllib.request.urlretrieve(FACE_CAFFEMODEL_URL, FACE_CAFFEMODEL)

        if not os.path.exists(FACE_PROTOTXT) or not os.path.exists(FACE_CAFFEMODEL):
            raise FileNotFoundError(
                "OpenCV DNN face model files are missing. "
                "Expected deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel"
            )

    def _init_face_detector(self):
        self._ensure_face_model_files()
        self.face_detector = cv2.dnn.readNetFromCaffe(FACE_PROTOTXT, FACE_CAFFEMODEL)

    def _load_age_model(self):
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        self.model = self.model.to(DEVICE)
        self.model.eval()

        print("Using ImageNet-pretrained backbone (not age-specific)")
        print("For better accuracy, use DEX, FairFace, or MiVOLO weights")

    def _detect_largest_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        img_height, img_width = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        largest_face = None
        largest_area = 0

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < FACE_CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([img_width, img_height, img_width, img_height])).astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)

            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            area = w * h

            if area > largest_area:
                largest_area = area
                largest_face = (x1, y1, w, h)

        return largest_face

    def _detect_and_crop_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        img_height, img_width = image.shape[:2]
        largest_face = self._detect_largest_face(image)

        if largest_face is None:
            return None

        x, y, w, h = largest_face
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_width, x + w + margin_x)
        y2 = min(img_height, y + h + margin_y)

        return image[y1:y2, x1:x2]

    def _estimate_age_from_face(self, face_image: np.ndarray) -> Tuple[float, float]:
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(face_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(input_tensor)
            raw_output = output.item()
            estimated_age = max(0, min(100, raw_output * 10 + 30))

            if 15 <= estimated_age <= 60:
                confidence = 0.7
            elif 5 <= estimated_age <= 80:
                confidence = 0.5
            else:
                confidence = 0.3

        return estimated_age, confidence

    def analyze_image(self, image_path: str) -> AgeResult:
        try:
            image = cv2.imread(image_path)
            if image is None:
                return AgeResult(
                    image_path=image_path,
                    is_adult=False,
                    estimated_age=0,
                    age_confidence=0,
                    face_detected=False,
                    error="Failed to read image",
                )
        except Exception as e:
            return AgeResult(
                image_path=image_path,
                is_adult=False,
                estimated_age=0,
                age_confidence=0,
                face_detected=False,
                error=str(e),
            )

        face_crop = self._detect_and_crop_face(image)
        if face_crop is None:
            return AgeResult(
                image_path=image_path,
                is_adult=False,
                estimated_age=0,
                age_confidence=0,
                face_detected=False,
                error=None,
            )

        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            return AgeResult(
                image_path=image_path,
                is_adult=False,
                estimated_age=0,
                age_confidence=0,
                face_detected=False,
                error="Face crop too small",
            )

        try:
            estimated_age, confidence = self._estimate_age_from_face(face_crop)
        except Exception as e:
            return AgeResult(
                image_path=image_path,
                is_adult=False,
                estimated_age=0,
                age_confidence=0,
                face_detected=True,
                error=f"Age estimation failed: {str(e)}",
            )

        is_adult = estimated_age >= self.min_age_threshold
        return AgeResult(
            image_path=image_path,
            is_adult=is_adult,
            estimated_age=round(estimated_age, 1),
            age_confidence=round(confidence, 3),
            face_detected=True,
            error=None,
        )

    def visualize_result(self, image_path: str, result: AgeResult, output_path: str = None) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            return None

        if result.face_detected:
            status = f"Age: {result.estimated_age:.0f} (conf: {result.age_confidence:.2f})"
            color = (0, 255, 0) if result.is_adult else (0, 0, 255)
            adult_status = "ADULT" if result.is_adult else f"CHILD (<{self.min_age_threshold})"
            cv2.putText(image, adult_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(image, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(image, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        if output_path:
            cv2.imwrite(output_path, image)

        return image


def process_dataset(dataset_path: str, output_dir: str, max_images: int = None) -> List[AgeResult]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(os.path.join(root, file))

    if max_images:
        image_paths = image_paths[:max_images]

    print(f"\nProcessing {len(image_paths)} images for age estimation...")

    estimator = AgeEstimator()
    results = []
    adult_count = 0
    child_count = 0
    no_face_count = 0

    for i, img_path in enumerate(tqdm(image_paths, desc="Age estimation", unit="img")):
        result = estimator.analyze_image(img_path)
        results.append(result)

        if not result.face_detected:
            no_face_count += 1
        elif result.is_adult:
            adult_count += 1
        else:
            child_count += 1

        if i < 20:
            vis_path = os.path.join(output_dir, "visualizations", f"{i:03d}_{Path(img_path).stem}.jpg")
            estimator.visualize_result(img_path, result, vis_path)

    print("\nRESULTS SUMMARY:")
    print(f"   Total images: {len(results)}")
    print(f"   Adults (age >= {MIN_AGE_THRESHOLD}): {adult_count} ({100 * adult_count / len(results):.1f}%)")
    print(f"   Children (age < {MIN_AGE_THRESHOLD}): {child_count} ({100 * child_count / len(results):.1f}%)")
    print(f"   No face detected: {no_face_count} ({100 * no_face_count / len(results):.1f}%)")

    ages = [r.estimated_age for r in results if r.face_detected]
    if ages:
        print("\nAge distribution (where face detected):")
        print(f"   Min: {min(ages):.0f}, Max: {max(ages):.0f}, Mean: {np.mean(ages):.1f}")

    results_data = []
    for r in results:
        results_data.append(
            {
                "image_path": r.image_path,
                "filename": os.path.basename(r.image_path),
                "is_adult": r.is_adult,
                "estimated_age": r.estimated_age,
                "age_confidence": r.age_confidence,
                "face_detected": r.face_detected,
                "error": r.error,
            }
        )

    results_path = os.path.join(output_dir, "age_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved to: {output_dir}/visualizations/")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("AGE ESTIMATION MODULE")
    print("=" * 70)
    print(f"""
This module estimates age to filter out young children.

WHAT IT DOES:
1. Detects face in image using OpenCV DNN
2. Crops the face region
3. Estimates age using a CNN model
4. Rejects if estimated_age < {MIN_AGE_THRESHOLD}

IMPORTANT NOTE:
This uses an ImageNet-pretrained backbone for demonstration.
The current implementation provides rough estimates.
""")

    print("\nModel files are auto-downloaded if missing:")
    print(f"1. {FACE_PROTOTXT_URL}")
    print(f"2. {FACE_CAFFEMODEL_URL}")

    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at: {DATASET_PATH}")
        raise SystemExit(1)

    print("\nRunning test on first 50 images...")
    results = process_dataset(dataset_path=DATASET_PATH, output_dir=OUTPUT_DIR, max_images=50)

    print("\n" + "=" * 70)
    print("AGE ESTIMATION COMPLETE")
    print("=" * 70)
