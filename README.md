Reduce manual dataset curation effort while enforcing these dataset requirements:

Full-body person crops only (no cropped feet/hands)
Face must be visible (frontal or side)
Exclude advertisements
Exclude young children (below teenager)
System Design
The pipeline uses a fail-fast multi-stage filter chain:

Face visibility check
Full-body visibility check
Age check
Advertisement check
If an image fails any stage, later stages are skipped. This keeps the system efficient and scalable.

Models and Techniques
No vision-language chat models are used.

Face detection: OpenCV DNN SSD face detector
Full-body detection: YOLOv8 pose keypoints (yolov8m-pose.pt)
Age estimation: ResNet18 regression head (ImageNet backbone)
Advertisement detection: CLIP zero-shot similarity scoring
Repository Structure
face_detection.py: standalone face module
fullbody_detection.py: standalone full-body module
age_estimation.py: standalone age module
ad_detection.py: standalone advertisement module
integrated_pipeline.py: end-to-end curation pipeline
evaluation.py: labeled-subset evaluation script
Unified Threshold Policy
The project now uses a unified teenager threshold of 13 across modules.

Integrated pipeline thresholds:

face_confidence: 0.3
face_min_area_ratio: 0.001
keypoint_confidence: 0.3
min_keypoints: 13
min_age: 13
ad_margin: 0.05
Important Reliability Policy
Age stage is fail-closed for unknown age:

If age inference fails, image is rejected with manual-review reason.
If age crop is invalid/unavailable, image is rejected with manual-review reason.
This prevents unknown-age images from being auto-accepted.

Data Paths
Main dataset path:

C:/Users/alank/OneDrive/Documents/image_filtering/image_filtering_pipeline/person-20260414T072836Z-3-001/person

Labeled subset path:

C:/Users/alank/OneDrive/Documents/image_filtering/image_filtering_pipeline/labeling_subset

OpenCV Face Model Assets
These files are auto-downloaded when missing:

https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
How to Run
Run end-to-end pipeline on main dataset:

python integrated_pipeline.py
Run evaluation on labeled subset predictions:

python evaluation.py
Latest Experimental Results (Labeled Subset)
Run date: 2026-04-17 Subset size: 100 images Ground truth distribution: 10 keep, 90 reject

Pipeline processing summary:

Accepted: 10/100 (10.0%)
Rejected: 90/100 (90.0%)
Mean latency: 326.2 ms/image
Total time: 32.6 s
Overall evaluation metrics:

Accuracy: 92.0% (meets >90% target)
Precision: 60.0%
Recall: 60.0%
F1-score: 60.0%
Confusion matrix counts:

TP: 6
FP: 4
TN: 86
FN: 4
Per-filter accuracy:

Full body: 88.0%
Face visible: 83.0%
Adult (age): 65.0%
Not advertisement: 25.0%
Main observed failure mode:

Remaining false negatives are still mostly "No face detected" at the face stage.
Next Improvements
Increase face recall (consider larger input scale or detector upgrade).
Improve age model quality with dedicated age-estimation weights.
Calibrate ad prompts/margin to improve not-advertisement performance.
Tune thresholds to reach >90% on labeled validation split while preserving generalization.
