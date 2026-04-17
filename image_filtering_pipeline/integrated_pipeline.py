"""
================================================================================
INTEGRATED IMAGE FILTERING PIPELINE
================================================================================
Purpose: Combine all 4 filters into a single pipeline that processes images
         and outputs a curated dataset meeting all requirements.

This version replaces MediaPipe face detection with OpenCV DNN.
================================================================================
"""

import os
import json
import time
import shutil
import urllib.request
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_PATH = r"C:\Users\alank\OneDrive\Documents\image_filtering\image_filtering_pipeline\person-20260414T072836Z-3-001\person"

OUTPUT_DIR = "pipeline_output"
ACCEPTED_DIR = os.path.join(OUTPUT_DIR, "accepted")
REJECTED_DIR = os.path.join(OUTPUT_DIR, "rejected")

COPY_IMAGES = True
MAX_IMAGES = None

THRESHOLDS = {
    "face_confidence": 0.3,
    "face_min_area_ratio": 0.001,
    "keypoint_confidence": 0.3,
    "min_keypoints": 13,
    "min_age": 13,
    "ad_margin": 0.05,
}

FACE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "opencv_face_detector")
FACE_PROTOTXT = os.path.join(FACE_MODEL_DIR, "deploy.prototxt")
FACE_CAFFEMODEL = os.path.join(FACE_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
FACE_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


@dataclass
class PipelineResult:
    image_path: str
    filename: str
    keep: bool
    rejection_reason: Optional[str]
    full_body_pass: bool
    face_visible_pass: bool
    is_adult_pass: bool
    not_advertisement_pass: bool
    keypoints_visible: int
    face_confidence: float
    face_area_ratio: float
    estimated_age: float
    ad_score: float
    natural_score: float
    processing_time_ms: float
    error: Optional[str] = None


class ImageFilteringPipeline:
    def __init__(self, thresholds: dict = None):
        print("=" * 70)
        print("INITIALIZING IMAGE FILTERING PIPELINE")
        print("=" * 70)

        self.thresholds = thresholds or THRESHOLDS
        self._init_modules()

        print("\nPipeline initialized and ready")
        print("=" * 70)

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

    def _init_modules(self):
        print("\n[1/4] Loading Full-Body Detector (YOLOv8-Pose)...")
        from ultralytics import YOLO

        self.pose_model = YOLO("yolov8m-pose.pt")
        print("      Loaded")

        print("\n[2/4] Loading Face Detector (OpenCV DNN)...")
        self._ensure_face_model_files()
        self.face_detector = cv2.dnn.readNetFromCaffe(FACE_PROTOTXT, FACE_CAFFEMODEL)
        print("      Loaded")

        print("\n[3/4] Loading Age Estimator...")
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"      Device: {self.device}")

        self.age_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.age_model.fc.in_features
        self.age_model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
        self.age_model = self.age_model.to(self.device)
        self.age_model.eval()

        self.age_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        print("      Loaded")

        print("\n[4/4] Loading Advertisement Detector (CLIP)...")
        try:
            import clip

            self.clip_module = clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

            ad_prompts = [
                "an advertisement",
                "a promotional photo",
                "a product advertisement",
                "a marketing image",
                "a commercial photograph",
                "a model in a fashion advertisement",
                "a stock photo for advertising",
            ]
            natural_prompts = [
                "a candid photo of a person",
                "a street photography portrait",
                "a natural photograph of someone",
                "a casual photo of a person",
                "a documentary photograph",
                "a photo of someone in daily life",
            ]

            with torch.no_grad():
                self.ad_text_features = self._encode_texts(ad_prompts)
                self.natural_text_features = self._encode_texts(natural_prompts)

            self.clip_available = True
            print("      Loaded")
        except ImportError:
            print("      CLIP not available - ad detection disabled")
            self.clip_available = False

    def _encode_texts(self, texts: List[str]):
        import torch

        text_tokens = self.clip_module.tokenize(texts).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_text(text_tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def _detect_faces(self, image: np.ndarray) -> List[dict]:
        img_height, img_width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.thresholds["face_confidence"]:
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
                    "confidence": confidence,
                    "area_ratio": area_ratio,
                }
            )

        faces.sort(key=lambda item: item["area_ratio"], reverse=True)
        return faces

    def process_image(self, image_path: str) -> PipelineResult:
        import torch

        start_time = time.time()
        filename = os.path.basename(image_path)

        result = {
            "image_path": image_path,
            "filename": filename,
            "keep": False,
            "rejection_reason": None,
            "full_body_pass": False,
            "face_visible_pass": False,
            "is_adult_pass": False,
            "not_advertisement_pass": False,
            "keypoints_visible": 0,
            "face_confidence": 0.0,
            "face_area_ratio": 0.0,
            "estimated_age": 0.0,
            "ad_score": 0.0,
            "natural_score": 0.0,
            "processing_time_ms": 0.0,
            "error": None,
        }

        try:
            image = cv2.imread(image_path)
            if image is None:
                result["error"] = "Failed to read image"
                result["processing_time_ms"] = (time.time() - start_time) * 1000
                return PipelineResult(**result)
        except Exception as e:
            result["error"] = str(e)
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            return PipelineResult(**result)

        img_height, img_width = image.shape[:2]

        # FILTER 1: FACE DETECTION
        faces = self._detect_faces(image)
        if not faces:
            result["rejection_reason"] = "No face detected"
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            return PipelineResult(**result)

        largest_face = faces[0]
        largest_conf = float(largest_face["confidence"])
        largest_area_ratio = float(largest_face["area_ratio"])

        result["face_confidence"] = round(largest_conf, 3)
        result["face_area_ratio"] = round(largest_area_ratio, 4)

        if largest_area_ratio < self.thresholds["face_min_area_ratio"]:
            result["rejection_reason"] = "Face too small (background person)"
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            return PipelineResult(**result)

        result["face_visible_pass"] = True

        # FILTER 2: FULL-BODY DETECTION
        pose_results = self.pose_model(image, verbose=False, conf=0.5)
        if pose_results[0].keypoints is None or len(pose_results[0].keypoints) == 0:
            result["rejection_reason"] = "No body keypoints detected"
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            return PipelineResult(**result)

        keypoints = pose_results[0].keypoints[0].data.cpu().numpy()[0]
        visible_kps = sum(1 for kp in keypoints if kp[2] >= self.thresholds["keypoint_confidence"])
        result["keypoints_visible"] = visible_kps

        has_hands = any(keypoints[i][2] >= self.thresholds["keypoint_confidence"] for i in [9, 10])
        has_feet = any(keypoints[i][2] >= self.thresholds["keypoint_confidence"] for i in [15, 16])
        has_core = all(keypoints[i][2] >= self.thresholds["keypoint_confidence"] for i in [5, 6, 11, 12])

        if not (visible_kps >= self.thresholds["min_keypoints"] and has_hands and has_feet and has_core):
            if not has_feet:
                result["rejection_reason"] = "Feet not visible (cropped)"
            elif not has_hands:
                result["rejection_reason"] = "Hands not visible (cropped)"
            elif not has_core:
                result["rejection_reason"] = "Core body not fully visible"
            else:
                result["rejection_reason"] = f"Only {visible_kps}/17 keypoints visible"

            result["processing_time_ms"] = (time.time() - start_time) * 1000
            return PipelineResult(**result)

        result["full_body_pass"] = True

        # FILTER 3: AGE ESTIMATION
        x = largest_face["x"]
        y = largest_face["y"]
        w = largest_face["width"]
        h = largest_face["height"]

        margin = int(min(w, h) * 0.2)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_width, x + w + margin)
        y2 = min(img_height, y + h + margin)

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size > 0 and face_crop.shape[0] > 10 and face_crop.shape[1] > 10:
            try:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                input_tensor = self.age_transform(face_rgb).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.age_model(input_tensor)
                    estimated_age = max(0, min(100, output.item() * 10 + 30))

                result["estimated_age"] = round(estimated_age, 1)

                if estimated_age < self.thresholds["min_age"]:
                    result["rejection_reason"] = f"Estimated age {estimated_age:.0f} < {self.thresholds['min_age']}"
                    result["processing_time_ms"] = (time.time() - start_time) * 1000
                    return PipelineResult(**result)

                result["is_adult_pass"] = True
            except Exception as e:
                result["estimated_age"] = -1
                result["rejection_reason"] = f"Age estimation failed (manual review): {str(e)}"
                result["processing_time_ms"] = (time.time() - start_time) * 1000
                return PipelineResult(**result)
        else:
            result["estimated_age"] = -1
            result["rejection_reason"] = "Age estimation unavailable (manual review)"
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            return PipelineResult(**result)

        # FILTER 4: ADVERTISEMENT DETECTION
        if self.clip_available:
            try:
                pil_image = Image.open(image_path).convert("RGB")
                image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    ad_sims = (image_features @ self.ad_text_features.T).squeeze(0).cpu().numpy()
                    natural_sims = (image_features @ self.natural_text_features.T).squeeze(0).cpu().numpy()

                ad_score = float(np.mean(ad_sims))
                natural_score = float(np.mean(natural_sims))
                result["ad_score"] = round(ad_score, 4)
                result["natural_score"] = round(natural_score, 4)

                if ad_score > natural_score + self.thresholds["ad_margin"]:
                    result["rejection_reason"] = "Detected as advertisement"
                    result["processing_time_ms"] = (time.time() - start_time) * 1000
                    return PipelineResult(**result)

                result["not_advertisement_pass"] = True
            except Exception:
                result["not_advertisement_pass"] = True
        else:
            result["not_advertisement_pass"] = True

        result["keep"] = True
        result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
        return PipelineResult(**result)

    def process_dataset(self, dataset_path: str, output_dir: str, max_images: int = None, copy_images: bool = True) -> List[PipelineResult]:
        os.makedirs(output_dir, exist_ok=True)
        accepted_dir = os.path.join(output_dir, "accepted")
        rejected_dir = os.path.join(output_dir, "rejected")

        if copy_images:
            os.makedirs(accepted_dir, exist_ok=True)
            os.makedirs(rejected_dir, exist_ok=True)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))

        if max_images:
            image_paths = image_paths[:max_images]

        print(f"\nProcessing {len(image_paths)} images through the pipeline...")
        print("=" * 70)

        results = []
        accepted_count = 0
        rejection_reasons = {}

        for img_path in tqdm(image_paths, desc="Processing", unit="img"):
            result = self.process_image(img_path)
            results.append(result)

            if result.keep:
                accepted_count += 1
                if copy_images:
                    shutil.copy2(img_path, os.path.join(accepted_dir, result.filename))
            else:
                reason = result.rejection_reason or "Unknown"
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                if copy_images:
                    shutil.copy2(img_path, os.path.join(rejected_dir, result.filename))

        print("\n" + "=" * 70)
        print("PIPELINE RESULTS SUMMARY")
        print("=" * 70)
        print(f"\n   Total images processed: {len(results)}")
        print(f"   Accepted: {accepted_count} ({100 * accepted_count / len(results):.1f}%)")
        print(f"   Rejected: {len(results) - accepted_count} ({100 * (len(results) - accepted_count) / len(results):.1f}%)")

        print("\n   Rejection breakdown:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"   - {reason}: {count} ({100 * count / len(results):.1f}%)")

        face_pass = sum(1 for r in results if r.face_visible_pass)
        body_pass = sum(1 for r in results if r.full_body_pass)
        age_pass = sum(1 for r in results if r.is_adult_pass)
        ad_pass = sum(1 for r in results if r.not_advertisement_pass)

        print("\n   Individual filter pass rates:")
        print(f"   - Face visible: {face_pass}/{len(results)} ({100 * face_pass / len(results):.1f}%)")
        print(f"   - Full body: {body_pass}/{len(results)} ({100 * body_pass / len(results):.1f}%)")
        print(f"   - Adult: {age_pass}/{len(results)} ({100 * age_pass / len(results):.1f}%)")
        print(f"   - Not advertisement: {ad_pass}/{len(results)} ({100 * ad_pass / len(results):.1f}%)")

        times = [r.processing_time_ms for r in results]
        print("\n   Processing speed:")
        print(f"   - Mean: {np.mean(times):.1f} ms/image")
        print(f"   - Total: {sum(times) / 1000:.1f} seconds")

        results_data = [asdict(r) for r in results]

        results_path = os.path.join(output_dir, "pipeline_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        csv_path = os.path.join(output_dir, "pipeline_results.csv")
        df = pd.DataFrame(results_data)
        df.to_csv(csv_path, index=False)

        print("\nResults saved to:")
        print(f"   - {results_path}")
        print(f"   - {csv_path}")

        if copy_images:
            print("\nImages copied to:")
            print(f"   - {accepted_dir}/ ({accepted_count} images)")
            print(f"   - {rejected_dir}/ ({len(results) - accepted_count} images)")

        return results


if __name__ == "__main__":
    print("=" * 70)
    print("INTEGRATED IMAGE FILTERING PIPELINE")
    print("=" * 70)
    print("""
This pipeline filters images based on ALL requirements:
1. Full-body visible (no cropped feet/hands)
2. Face visible (frontal or side view)
3. Adult (age >= 13)
4. Not an advertisement

Images must pass ALL filters to be accepted.
""")

    print("\nFace model files are auto-downloaded if missing:")
    print(f"1. {FACE_PROTOTXT_URL}")
    print(f"2. {FACE_CAFFEMODEL_URL}")

    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at: {DATASET_PATH}")
        raise SystemExit(1)

    pipeline = ImageFilteringPipeline(thresholds=THRESHOLDS)
    results = pipeline.process_dataset(
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        max_images=MAX_IMAGES,
        copy_images=COPY_IMAGES,
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
