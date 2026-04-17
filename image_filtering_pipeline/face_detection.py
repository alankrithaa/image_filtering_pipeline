"""
================================================================================
FACE VISIBILITY MODULE (Task 2.2)
================================================================================
Purpose: Detect if a face (frontal or side view) is visible in the image.

This version uses OpenCV DNN face detector (SSD ResNet-10) instead of MediaPipe
for compatibility with newer MediaPipe releases.
================================================================================
"""

import os
import cv2
import json
import numpy as np
import urllib.request
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_PATH = r"C:\Users\alank\OneDrive\Documents\image_filtering\image_filtering_pipeline\person-20260414T072836Z-3-001\person"
OUTPUT_DIR = "face_results"

FACE_CONFIDENCE_THRESHOLD = 0.5
MIN_FACE_AREA_RATIO = 0.005  # 0.5%
MAX_FACES = 5

FACE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "opencv_face_detector")
FACE_PROTOTXT = os.path.join(FACE_MODEL_DIR, "deploy.prototxt")
FACE_CAFFEMODEL = os.path.join(FACE_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
FACE_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


@dataclass
class FaceResult:
    image_path: str
    has_visible_face: bool
    num_faces_detected: int
    largest_face_confidence: float
    largest_face_area_ratio: float
    face_locations: List[dict]
    error: Optional[str] = None


class FaceDetector:
    """Detects faces in images using OpenCV DNN face detector."""

    def __init__(
        self,
        confidence_threshold: float = FACE_CONFIDENCE_THRESHOLD,
        min_face_area_ratio: float = MIN_FACE_AREA_RATIO,
    ):
        print("Initializing OpenCV DNN Face Detection...")
        self.confidence_threshold = confidence_threshold
        self.min_face_area_ratio = min_face_area_ratio
        self._ensure_model_files()
        self.face_net = cv2.dnn.readNetFromCaffe(FACE_PROTOTXT, FACE_CAFFEMODEL)
        print("Face Detection initialized")

    def _ensure_model_files(self):
        """Ensure model files exist; download if missing."""
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

    def _detect_faces(self, image: np.ndarray) -> List[dict]:
        img_height, img_width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([img_width, img_height, img_width, img_height])).astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)

            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w == 0 or h == 0:
                continue

            area_ratio = (w * h) / float(img_width * img_height)
            faces.append(
                {
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(w),
                    "height": int(h),
                    "confidence": round(confidence, 3),
                    "area_ratio": round(area_ratio, 4),
                }
            )

        faces.sort(key=lambda item: item["area_ratio"], reverse=True)
        return faces[:MAX_FACES]

    def analyze_image(self, image_path: str) -> FaceResult:
        try:
            image = cv2.imread(image_path)
            if image is None:
                return FaceResult(
                    image_path=image_path,
                    has_visible_face=False,
                    num_faces_detected=0,
                    largest_face_confidence=0.0,
                    largest_face_area_ratio=0.0,
                    face_locations=[],
                    error="Failed to read image",
                )
        except Exception as e:
            return FaceResult(
                image_path=image_path,
                has_visible_face=False,
                num_faces_detected=0,
                largest_face_confidence=0.0,
                largest_face_area_ratio=0.0,
                face_locations=[],
                error=str(e),
            )

        face_locations = self._detect_faces(image)
        if not face_locations:
            return FaceResult(
                image_path=image_path,
                has_visible_face=False,
                num_faces_detected=0,
                largest_face_confidence=0.0,
                largest_face_area_ratio=0.0,
                face_locations=[],
                error=None,
            )

        largest_face = max(face_locations, key=lambda item: item["area_ratio"])
        largest_area_ratio = float(largest_face["area_ratio"])
        largest_confidence = float(largest_face["confidence"])

        has_visible_face = (
            len(face_locations) > 0
            and largest_area_ratio >= self.min_face_area_ratio
            and largest_confidence >= self.confidence_threshold
        )

        return FaceResult(
            image_path=image_path,
            has_visible_face=has_visible_face,
            num_faces_detected=len(face_locations),
            largest_face_confidence=round(largest_confidence, 3),
            largest_face_area_ratio=round(largest_area_ratio, 4),
            face_locations=face_locations,
            error=None,
        )

    def detect(self, image_path: str) -> dict:
        """Expected API shape for integrations that use dictionary output."""
        result = self.analyze_image(image_path)

        reason = "ok"
        if result.error:
            reason = result.error
        elif not result.has_visible_face and result.num_faces_detected == 0:
            reason = "no_face_detected"
        elif result.largest_face_area_ratio < self.min_face_area_ratio:
            reason = "face_too_small"
        elif result.largest_face_confidence < self.confidence_threshold:
            reason = "low_confidence"

        return {
            "face_visible": result.has_visible_face,
            "num_faces": result.num_faces_detected,
            "largest_face_ratio": result.largest_face_area_ratio,
            "confidence": result.largest_face_confidence,
            "faces": result.face_locations,
            "reason": reason,
        }

    def visualize_result(self, image_path: str, result: FaceResult, output_path: str = None) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            return None

        for face in result.face_locations:
            x, y, w, h = face["x"], face["y"], face["width"], face["height"]
            conf = face["confidence"]
            area_pct = face["area_ratio"] * 100

            passes = face["area_ratio"] >= self.min_face_area_ratio and face["confidence"] >= self.confidence_threshold
            color = (0, 255, 0) if passes else (0, 165, 255)

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            label = f"Conf: {conf:.2f} | Area: {area_pct:.1f}%"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        status = "FACE VISIBLE" if result.has_visible_face else "NO VALID FACE"
        status_color = (0, 255, 0) if result.has_visible_face else (0, 0, 255)
        cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        if output_path:
            cv2.imwrite(output_path, image)

        return image

    def visualize(self, image_path: str, result_dict: dict, output_path: str = None) -> np.ndarray:
        """Expected API shape for integrations that call visualize()."""
        face_result = FaceResult(
            image_path=image_path,
            has_visible_face=bool(result_dict.get("face_visible", False)),
            num_faces_detected=int(result_dict.get("num_faces", 0)),
            largest_face_confidence=float(result_dict.get("confidence", 0.0)),
            largest_face_area_ratio=float(result_dict.get("largest_face_ratio", 0.0)),
            face_locations=list(result_dict.get("faces", [])),
            error=None if result_dict.get("reason") == "ok" else result_dict.get("reason"),
        )
        return self.visualize_result(image_path, face_result, output_path)


def process_dataset(dataset_path: str, output_dir: str, max_images: int = None) -> List[FaceResult]:
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

    print(f"\nProcessing {len(image_paths)} images for face detection...")

    detector = FaceDetector()
    results = []
    face_visible_count = 0

    for i, img_path in enumerate(tqdm(image_paths, desc="Face detection", unit="img")):
        result = detector.analyze_image(img_path)
        results.append(result)

        if result.has_visible_face:
            face_visible_count += 1

        if i < 20:
            vis_path = os.path.join(output_dir, "visualizations", f"{i:03d}_{Path(img_path).stem}.jpg")
            detector.visualize_result(img_path, result, vis_path)

    print("\nRESULTS SUMMARY:")
    print(f"   Total images: {len(results)}")
    print(f"   Face visible: {face_visible_count} ({100 * face_visible_count / len(results):.1f}%)")
    print(f"   No valid face: {len(results) - face_visible_count} ({100 * (len(results) - face_visible_count) / len(results):.1f}%)")

    faces_detected = [r.num_faces_detected for r in results if not r.error]
    avg_faces = np.mean(faces_detected) if faces_detected else 0
    print(f"   Average faces per image: {avg_faces:.2f}")

    results_data = []
    for r in results:
        results_data.append(
            {
                "image_path": r.image_path,
                "filename": os.path.basename(r.image_path),
                "has_visible_face": r.has_visible_face,
                "num_faces_detected": r.num_faces_detected,
                "largest_face_confidence": r.largest_face_confidence,
                "largest_face_area_ratio": r.largest_face_area_ratio,
                "face_locations": r.face_locations,
                "error": r.error,
            }
        )

    results_path = os.path.join(output_dir, "face_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved to: {output_dir}/visualizations/")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("FACE VISIBILITY MODULE")
    print("=" * 70)
    print("""
This module uses OpenCV DNN Face Detection to verify face visibility.

WHAT IT CHECKS:
1. At least one face detected in the image
2. Face is large enough (not a background person)
3. Detection confidence is high enough

WHAT IT REJECTS:
1. Images with no detectable face
2. Images where person is facing away (back of head)
3. Images where face is too small (crowd scenes, distant people)
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
    print("FACE DETECTION COMPLETE")
    print("=" * 70)
