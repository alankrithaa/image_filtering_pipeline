"""
MediaPipe Diagnostic Script
Run this to check if MediaPipe is installed correctly.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Check for conflicting files
import os
current_dir = os.getcwd()
py_files = [f for f in os.listdir(current_dir) if f.endswith('.py')]
conflicts = [f for f in py_files if f.lower() in ['mediapipe.py', 'cv2.py', 'cv.py', 'numpy.py', 'torch.py']]

if conflicts:
    print("⚠️  WARNING: Found potentially conflicting files in your directory:")
    for f in conflicts:
        print(f"   - {f}")
    print("   These may shadow the actual library imports!")
    print("   Please rename these files (e.g., add 'my_' prefix)")
    print()

# Try importing mediapipe
print("Testing MediaPipe import...")
try:
    import mediapipe
    print(f"✅ mediapipe imported successfully")
    print(f"   Version: {mediapipe.__version__}")
    print(f"   Location: {mediapipe.__file__}")
    
    # Check if solutions exists
    if hasattr(mediapipe, 'solutions'):
        print("✅ mediapipe.solutions is available")
        
        # Check face detection
        if hasattr(mediapipe.solutions, 'face_detection'):
            print("✅ mediapipe.solutions.face_detection is available")
            
            # Try to create a detector
            mp_face = mediapipe.solutions.face_detection
            detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            print("✅ FaceDetection object created successfully")
            detector.close()
        else:
            print("❌ mediapipe.solutions.face_detection NOT found")
    else:
        print("❌ mediapipe.solutions NOT found")
        print()
        print("This usually means:")
        print("1. There's a file named 'mediapipe.py' in your directory")
        print("2. MediaPipe is corrupted - try reinstalling:")
        print("   pip uninstall mediapipe -y")
        print("   pip install mediapipe")
        
except ImportError as e:
    print(f"❌ Failed to import mediapipe: {e}")
    print()
    print("Install with: pip install mediapipe")

except Exception as e:
    print(f"❌ Error: {e}")

print()
print("=" * 50)

# Also check other dependencies
print("\nChecking other dependencies...")

deps = ['cv2', 'numpy', 'torch', 'ultralytics', 'tqdm', 'PIL']
for dep in deps:
    try:
        if dep == 'cv2':
            import cv2
            print(f"✅ {dep} (OpenCV) - version {cv2.__version__}")
        elif dep == 'numpy':
            import numpy
            print(f"✅ {dep} - version {numpy.__version__}")
        elif dep == 'torch':
            import torch
            print(f"✅ {dep} - version {torch.__version__}")
        elif dep == 'ultralytics':
            import ultralytics
            print(f"✅ {dep} - version {ultralytics.__version__}")
        elif dep == 'tqdm':
            import tqdm
            print(f"✅ {dep} - version {tqdm.__version__}")
        elif dep == 'PIL':
            from PIL import Image
            import PIL
            print(f"✅ {dep} (Pillow) - version {PIL.__version__}")
    except ImportError:
        print(f"❌ {dep} - NOT INSTALLED")

# Check CLIP
print()
try:
    import clip
    print(f"✅ CLIP installed")
except ImportError:
    print("❌ CLIP - NOT INSTALLED")
    print("   Install with: pip install git+https://github.com/openai/CLIP.git")
