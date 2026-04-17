# Image Filtering Pipeline

Automated dataset curation pipeline for person crops.

## Assignment Goal
Reduce manual dataset curation effort while enforcing these dataset requirements:

1. Full-body person crops only (no cropped feet/hands)
2. Face must be visible (frontal or side)
3. Exclude advertisements
4. Exclude young children (below teenager)

## System Design
The pipeline uses a fail-fast multi-stage filter chain:

1. Face visibility check
2. Full-body visibility check
3. Age check
4. Advertisement check

If an image fails any stage, later stages are skipped. This keeps the system efficient and scalable.

## Models and Techniques
No vision-language chat models are used.

1. Face detection: OpenCV DNN SSD face detector
2. Full-body detection: YOLOv8 pose keypoints (yolov8m-pose.pt)
3. Age estimation: ResNet18 regression head (ImageNet backbone)
4. Advertisement detection: CLIP zero-shot similarity scoring

## Repository Structure
1. face_detection.py: standalone face module
2. fullbody_detection.py: standalone full-body module
3. age_estimation.py: standalone age module
4. ad_detection.py: standalone advertisement module
5. integrated_pipeline.py: end-to-end curation pipeline
6. evaluation.py: labeled-subset evaluation script

## Unified Threshold Policy
The project now uses a unified teenager threshold of 13 across modules.

Integrated pipeline thresholds:

1. face_confidence: 0.3
2. face_min_area_ratio: 0.001
3. keypoint_confidence: 0.3
4. min_keypoints: 13
5. min_age: 13
6. ad_margin: 0.05

## Important Reliability Policy
Age stage is fail-closed for unknown age:

1. If age inference fails, image is rejected with manual-review reason.
2. If age crop is invalid/unavailable, image is rejected with manual-review reason.

This prevents unknown-age images from being auto-accepted.

## Data Paths
Main dataset path:

C:/Users/alank/OneDrive/Documents/image_filtering/image_filtering_pipeline/person-20260414T072836Z-3-001/person

Labeled subset path:

C:/Users/alank/OneDrive/Documents/image_filtering/image_filtering_pipeline/labeling_subset

## OpenCV Face Model Assets
These files are auto-downloaded when missing:

1. https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
2. https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

## How to Run
Run end-to-end pipeline on main dataset:

```powershell
python integrated_pipeline.py
```

Run evaluation on labeled subset predictions:

```powershell
python evaluation.py
```

## Latest Experimental Results (Labeled Subset)
Run date: 2026-04-17
Subset size: 100 images
Ground truth distribution: 10 keep, 90 reject

Pipeline processing summary:

1. Accepted: 10/100 (10.0%)
2. Rejected: 90/100 (90.0%)
3. Mean latency: 326.2 ms/image
4. Total time: 32.6 s

Overall evaluation metrics:

1. Accuracy: 92.0% (meets >90% target)
2. Precision: 60.0%
3. Recall: 60.0%
4. F1-score: 60.0%

Confusion matrix counts:

1. TP: 6
2. FP: 4
3. TN: 86
4. FN: 4

Per-filter accuracy:

1. Full body: 88.0%
2. Face visible: 83.0%
3. Adult (age): 65.0%
4. Not advertisement: 25.0%

Main observed failure mode:

1. Remaining false negatives are still mostly "No face detected" at the face stage.

## Next Improvements
1. Increase face recall (consider larger input scale or detector upgrade).
2. Improve age model quality with dedicated age-estimation weights.
3. Calibrate ad prompts/margin to improve not-advertisement performance.
4. Tune thresholds to reach >90% on labeled validation split while preserving generalization.