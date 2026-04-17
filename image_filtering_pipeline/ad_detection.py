"""
================================================================================
ADVERTISEMENT DETECTION MODULE (Task 2.4)
================================================================================
Purpose: Detect if an image is an advertisement/promotional photo.

WHY CLIP?
=========
CLIP (Contrastive Language-Image Pre-training) by OpenAI is perfect for this:
- ZERO-SHOT: No training data needed! Just describe what you're looking for
- FLEXIBLE: Compare images to ANY text description
- ROBUST: Trained on 400M image-text pairs from the internet

HOW IT WORKS:
=============
CLIP learns to map images and text into the same "embedding space".
Similar images and text end up close together in this space.

For advertisement detection:
1. Encode the image into an embedding vector
2. Encode text prompts like "an advertisement" into embedding vectors
3. Calculate similarity between image and each text prompt
4. If image is more similar to "advertisement" than "candid photo" → reject

TEXT PROMPTS STRATEGY:
======================
We compare against two groups of prompts:

AD PROMPTS (things we want to reject):
- "an advertisement"
- "a promotional photo"
- "a product advertisement"
- "a marketing image"
- "a commercial photograph"
- "a model in a fashion advertisement"

NATURAL PROMPTS (things we want to keep):
- "a candid photo of a person"
- "a street photography portrait"
- "a natural photograph of someone"
- "a casual photo of a person"

DECISION LOGIC:
===============
ad_score = average similarity to ad prompts
natural_score = average similarity to natural prompts

If ad_score > natural_score + margin → ADVERTISEMENT (reject)
Else → NATURAL PHOTO (keep)

================================================================================
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import json
import torch
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to your dataset - UPDATE THIS
DATASET_PATH = r"C:\Users\alank\OneDrive\Documents\image_filtering\image_filtering_pipeline\person-20260414T072836Z-3-001"

# Output directory for results
OUTPUT_DIR = "ad_results"

# Decision margin: how much higher must ad_score be than natural_score
# Higher margin = more lenient (fewer false positives, might miss some ads)
# Lower margin = more strict (catch more ads, might have false positives)
DECISION_MARGIN = 0.05

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Text prompts for classification
AD_PROMPTS = [
    "an advertisement",
    "a promotional photo",
    "a product advertisement", 
    "a marketing image",
    "a commercial photograph",
    "a model in a fashion advertisement",
    "a stock photo for advertising",
    "a professional advertisement photo",
    "a photo with text overlay",
    "a photo with a logo or brand",
]

NATURAL_PROMPTS = [
    "a candid photo of a person",
    "a street photography portrait",
    "a natural photograph of someone",
    "a casual photo of a person",
    "a documentary photograph",
    "a photo of someone in daily life",
    "a genuine moment captured",
    "a real person in a natural setting",
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AdResult:
    """
    Stores the result of advertisement detection for one image.
    """
    image_path: str
    is_advertisement: bool
    ad_score: float  # Average similarity to ad prompts
    natural_score: float  # Average similarity to natural prompts
    score_difference: float  # ad_score - natural_score
    top_ad_prompt: str  # Which ad prompt matched best
    top_ad_similarity: float
    error: Optional[str] = None


# =============================================================================
# CLIP-BASED ADVERTISEMENT DETECTOR
# =============================================================================

class AdvertisementDetector:
    """
    Detects advertisements using CLIP zero-shot classification.
    
    CLIP compares images to text descriptions. We compare each image
    to "advertisement" prompts vs "natural photo" prompts and see
    which group it matches better.
    
    Usage:
        detector = AdvertisementDetector()
        result = detector.analyze_image("path/to/image.jpg")
        print(result.is_advertisement)  # True or False
    """
    
    def __init__(self, decision_margin: float = DECISION_MARGIN):
        """
        Initialize CLIP model for advertisement detection.
        """
        print("🔄 Initializing CLIP for advertisement detection...")
        print(f"   Device: {DEVICE}")
        
        self.decision_margin = decision_margin
        
        # Try to import CLIP
        try:
            import clip
            self.clip = clip
        except ImportError:
            print("\n⚠️  CLIP not installed. Installing now...")
            print("   Run: pip install git+https://github.com/openai/CLIP.git")
            print("   Or:  pip install clip-by-openai")
            raise ImportError(
                "CLIP is required. Install with: "
                "pip install git+https://github.com/openai/CLIP.git"
            )
        
        # Load CLIP model
        # Options: "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"
        # ViT-B/32 is fastest, ViT-L/14 is most accurate
        print("   Loading CLIP model (ViT-B/32)...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=DEVICE)
        
        # Pre-encode text prompts (only need to do this once)
        print("   Encoding text prompts...")
        self.ad_text_features = self._encode_texts(AD_PROMPTS)
        self.natural_text_features = self._encode_texts(NATURAL_PROMPTS)
        
        print("✅ Advertisement Detector initialized!")
    
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of text prompts into CLIP embeddings.
        
        Args:
            texts: List of text prompts
            
        Returns:
            Tensor of shape (len(texts), embedding_dim)
        """
        text_tokens = self.clip.tokenize(texts).to(DEVICE)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize for cosine similarity
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def _encode_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Encode an image into a CLIP embedding.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tensor of shape (1, embedding_dim) or None if failed
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(DEVICE)
            
            # Encode
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features
            
        except Exception as e:
            print(f"   Error encoding image: {e}")
            return None
    
    def analyze_image(self, image_path: str) -> AdResult:
        """
        Analyze a single image for advertisement content.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            AdResult with detection details
        """
        # Encode image
        image_features = self._encode_image(image_path)
        
        if image_features is None:
            return AdResult(
                image_path=image_path,
                is_advertisement=False,
                ad_score=0.0,
                natural_score=0.0,
                score_difference=0.0,
                top_ad_prompt="",
                top_ad_similarity=0.0,
                error="Failed to encode image"
            )
        
        # Calculate similarities to ad prompts
        ad_similarities = (image_features @ self.ad_text_features.T).squeeze(0)
        ad_similarities = ad_similarities.cpu().numpy()
        
        # Calculate similarities to natural prompts
        natural_similarities = (image_features @ self.natural_text_features.T).squeeze(0)
        natural_similarities = natural_similarities.cpu().numpy()
        
        # Aggregate scores
        ad_score = float(np.mean(ad_similarities))
        natural_score = float(np.mean(natural_similarities))
        score_diff = ad_score - natural_score
        
        # Find top matching ad prompt
        top_ad_idx = int(np.argmax(ad_similarities))
        top_ad_prompt = AD_PROMPTS[top_ad_idx]
        top_ad_similarity = float(ad_similarities[top_ad_idx])
        
        # Decision: is this an advertisement?
        # If ad_score is significantly higher than natural_score, it's an ad
        is_advertisement = score_diff > self.decision_margin
        
        return AdResult(
            image_path=image_path,
            is_advertisement=is_advertisement,
            ad_score=round(ad_score, 4),
            natural_score=round(natural_score, 4),
            score_difference=round(score_diff, 4),
            top_ad_prompt=top_ad_prompt,
            top_ad_similarity=round(top_ad_similarity, 4),
            error=None
        )
    
    def visualize_result(self, image_path: str, result: AdResult,
                         output_path: str = None) -> np.ndarray:
        """
        Create a visualization showing ad detection result.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Status text
        if result.is_advertisement:
            status = "ADVERTISEMENT"
            color = (0, 0, 255)  # Red
        else:
            status = "NATURAL PHOTO"
            color = (0, 255, 0)  # Green
        
        cv2.putText(image, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Scores
        scores = f"Ad: {result.ad_score:.3f} | Natural: {result.natural_score:.3f}"
        cv2.putText(image, scores, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Top matching prompt
        prompt_text = f"Top match: {result.top_ad_prompt[:40]}"
        cv2.putText(image, prompt_text, (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_dataset(dataset_path: str, output_dir: str,
                    max_images: int = None) -> List[AdResult]:
    """
    Process all images for advertisement detection.
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
    
    print(f"\n📊 Processing {len(image_paths)} images for advertisement detection...")
    
    # Initialize detector
    detector = AdvertisementDetector()
    
    # Process images
    results = []
    ad_count = 0
    natural_count = 0
    error_count = 0
    
    for i, img_path in enumerate(image_paths):
        # Progress update
        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f"   Processed {i + 1}/{len(image_paths)} images "
                  f"({100*(i+1)/len(image_paths):.1f}%)")
        
        # Analyze image
        result = detector.analyze_image(img_path)
        results.append(result)
        
        if result.error:
            error_count += 1
        elif result.is_advertisement:
            ad_count += 1
        else:
            natural_count += 1
        
        # Save visualization for first 20 images
        if i < 20:
            vis_path = os.path.join(output_dir, "visualizations",
                                   f"{i:03d}_{Path(img_path).stem}.jpg")
            detector.visualize_result(img_path, result, vis_path)
    
    # Print summary
    print(f"\n📈 RESULTS SUMMARY:")
    print(f"   Total images: {len(results)}")
    print(f"   Natural photos: {natural_count} ({100*natural_count/len(results):.1f}%)")
    print(f"   Advertisements: {ad_count} ({100*ad_count/len(results):.1f}%)")
    print(f"   Errors: {error_count} ({100*error_count/len(results):.1f}%)")
    
    # Score distribution
    ad_scores = [r.ad_score for r in results if not r.error]
    natural_scores = [r.natural_score for r in results if not r.error]
    if ad_scores:
        print(f"\n   Score statistics:")
        print(f"   Ad scores - Mean: {np.mean(ad_scores):.3f}, Std: {np.std(ad_scores):.3f}")
        print(f"   Natural scores - Mean: {np.mean(natural_scores):.3f}, Std: {np.std(natural_scores):.3f}")
    
    # Top ad prompts that matched
    print(f"\n   Most common ad prompt matches:")
    prompt_counts = {}
    for r in results:
        if r.is_advertisement and r.top_ad_prompt:
            prompt_counts[r.top_ad_prompt] = prompt_counts.get(r.top_ad_prompt, 0) + 1
    for prompt, count in sorted(prompt_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"   - {prompt}: {count}")
    
    # Save results to JSON
    results_data = []
    for r in results:
        results_data.append({
            'image_path': r.image_path,
            'filename': os.path.basename(r.image_path),
            'is_advertisement': r.is_advertisement,
            'ad_score': r.ad_score,
            'natural_score': r.natural_score,
            'score_difference': r.score_difference,
            'top_ad_prompt': r.top_ad_prompt,
            'top_ad_similarity': r.top_ad_similarity,
            'error': r.error
        })
    
    results_path = os.path.join(output_dir, "ad_results.json")
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
    print("📢 ADVERTISEMENT DETECTION MODULE")
    print("="*70)
    print("""
This module uses CLIP to detect advertisement/promotional images.

HOW IT WORKS:
1. Encode image using CLIP vision encoder
2. Compare to "advertisement" text prompts
3. Compare to "natural photo" text prompts  
4. If more similar to ads → classify as advertisement

AD PROMPTS (reject if matched):
- "an advertisement"
- "a promotional photo"
- "a product advertisement"
- "a marketing image"
- etc.

NATURAL PROMPTS (keep if matched):
- "a candid photo of a person"
- "a street photography portrait"
- "a natural photograph"
- etc.
    """)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at: {DATASET_PATH}")
        print("   Please update DATASET_PATH in this script.")
        exit(1)
    
    # Check for CLIP
    try:
        import clip
    except ImportError:
        print("❌ CLIP not installed!")
        print("\nInstall with ONE of these commands:")
        print("   pip install git+https://github.com/openai/CLIP.git")
        print("   pip install clip-by-openai")
        print("   pip install openai-clip")
        exit(1)
    
    # Process dataset
    print("\n🧪 Running test on first 50 images...")
    print("   (Remove max_images parameter to process full dataset)\n")
    
    results = process_dataset(
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        max_images=50  # Remove this line to process all images
    )
    
    print("\n" + "="*70)
    print("✅ ADVERTISEMENT DETECTION COMPLETE!")
    print("="*70)
    print(f"""
NEXT STEPS:
1. Check the 'visualizations/' folder to verify detections look correct
2. Adjust DECISION_MARGIN if needed (currently {DECISION_MARGIN}):
   - Higher = fewer false positives (might miss some ads)
   - Lower = catch more ads (might have false positives)
3. Add/modify prompts in AD_PROMPTS and NATURAL_PROMPTS if needed
4. Integrate this module into your main pipeline
    """)
