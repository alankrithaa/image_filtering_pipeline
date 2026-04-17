"""
================================================================================
DATASET EXPLORATION SCRIPT FOR COMPUTER VISION
================================================================================


"""

import os
import random
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION - CHANGE THIS PATH TO YOUR DATASET
# =============================================================================
DATASET_PATH = r"C:\Users\alank\OneDrive\Documents\image_filtering\image_filtering_pipeline\person-20260414T072836Z-3-001"

# Number of random images to display for visual inspection
NUM_SAMPLES_TO_VIEW = 20

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def get_all_image_paths(dataset_path: str) -> list:
    """
    Recursively find all image files in the dataset folder.
    
    Why recursive? Datasets often have subfolders like:
    - /train, /test, /val
    - /class1, /class2
    - Or just a flat folder of images
    
    Args:
        dataset_path: Root folder containing images
        
    Returns:
        List of full paths to all image files
    """
    image_paths = []
    
    # os.walk traverses all subdirectories
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # Check if file extension is an image type
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    
    return image_paths


def analyze_image_sizes(image_paths: list) -> dict:
    """
    Analyze the dimensions of all images in the dataset.
    
    Why this matters for CV:
    - Neural networks often require fixed input sizes (e.g., 224x224, 640x640)
    - Very small images may lack detail for detection
    - Very large images slow down processing
    - Aspect ratios tell you if images are portraits, landscapes, or squares
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Dictionary with size statistics
    """
    widths = []
    heights = []
    aspect_ratios = []
    file_sizes_kb = []
    corrupted_files = []
    
    print("\n📊 Analyzing image sizes (this may take a moment)...")
    
    for i, path in enumerate(image_paths):
        # Progress indicator every 100 images
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(image_paths)} images...")
        
        try:
            # Use PIL to get dimensions (faster than cv2 for just reading metadata)
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
            
            # Get file size in KB
            file_size = os.path.getsize(path) / 1024
            file_sizes_kb.append(file_size)
            
        except Exception as e:
            # Some files may be corrupted or not actually images
            corrupted_files.append((path, str(e)))
    
    return {
        'widths': widths,
        'heights': heights,
        'aspect_ratios': aspect_ratios,
        'file_sizes_kb': file_sizes_kb,
        'corrupted_files': corrupted_files
    }


def print_statistics(image_paths: list, size_analysis: dict):
    """
    Print comprehensive statistics about the dataset.
    
    This gives you a "feel" for the data without looking at every image.
    """
    print("\n" + "="*60)
    print("📈 DATASET STATISTICS")
    print("="*60)
    
    # Basic counts
    print(f"\n📁 Total images found: {len(image_paths)}")
    print(f"❌ Corrupted/unreadable files: {len(size_analysis['corrupted_files'])}")
    
    if size_analysis['corrupted_files']:
        print("\n   Corrupted files:")
        for path, error in size_analysis['corrupted_files'][:5]:  # Show first 5
            print(f"   - {os.path.basename(path)}: {error}")
        if len(size_analysis['corrupted_files']) > 5:
            print(f"   ... and {len(size_analysis['corrupted_files']) - 5} more")
    
    # Dimension statistics
    widths = size_analysis['widths']
    heights = size_analysis['heights']
    
    if widths:
        print(f"\n📐 IMAGE DIMENSIONS:")
        print(f"   Width  - Min: {min(widths)}px, Max: {max(widths)}px, "
              f"Mean: {np.mean(widths):.0f}px, Median: {np.median(widths):.0f}px")
        print(f"   Height - Min: {min(heights)}px, Max: {max(heights)}px, "
              f"Mean: {np.mean(heights):.0f}px, Median: {np.median(heights):.0f}px")
        
        # Aspect ratio analysis
        aspects = size_analysis['aspect_ratios']
        portrait = sum(1 for a in aspects if a < 0.9)      # Taller than wide
        landscape = sum(1 for a in aspects if a > 1.1)     # Wider than tall
        square = sum(1 for a in aspects if 0.9 <= a <= 1.1)  # Roughly square
        
        print(f"\n📷 ASPECT RATIO DISTRIBUTION:")
        print(f"   Portrait (tall):   {portrait} ({100*portrait/len(aspects):.1f}%)")
        print(f"   Landscape (wide):  {landscape} ({100*landscape/len(aspects):.1f}%)")
        print(f"   Square-ish:        {square} ({100*square/len(aspects):.1f}%)")
        
        # File size statistics
        sizes = size_analysis['file_sizes_kb']
        print(f"\n💾 FILE SIZES:")
        print(f"   Min: {min(sizes):.1f} KB, Max: {max(sizes):.1f} KB")
        print(f"   Mean: {np.mean(sizes):.1f} KB, Median: {np.median(sizes):.1f} KB")
        print(f"   Total dataset size: {sum(sizes)/1024:.1f} MB")
    
    # Folder structure
    print(f"\n📂 FOLDER STRUCTURE:")
    folders = Counter(os.path.dirname(p) for p in image_paths)
    for folder, count in folders.most_common(10):
        print(f"   {folder}: {count} images")


def display_random_samples(image_paths: list, num_samples: int = 20):
    """
    Display a grid of random images for visual inspection.
    
    Why this is crucial:
    - Statistics don't show you what "noise" looks like
    - You need to SEE examples of:
      - Cropped bodies (missing feet, hands)
      - Obscured faces
      - Children
      - Advertisements
    - This builds your intuition for what filters you need
    
    Args:
        image_paths: List of all image paths
        num_samples: How many random images to display
    """
    print(f"\n🖼️  Displaying {num_samples} random samples...")
    print("   Close the window to continue.\n")
    
    # Randomly select images
    samples = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    # Calculate grid dimensions (aim for roughly square grid)
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, (ax, img_path) in enumerate(zip(axes, samples)):
        try:
            # Read image with OpenCV (reads as BGR)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Convert BGR to RGB for matplotlib
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                
                # Show filename (truncated) and dimensions
                filename = os.path.basename(img_path)
                if len(filename) > 15:
                    filename = filename[:12] + "..."
                ax.set_title(f"{filename}\n{img.shape[1]}x{img.shape[0]}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "Failed to load", ha='center', va='center')
                
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)[:20]}", ha='center', va='center')
        
        ax.axis('off')
    
    # Hide any unused subplots
    for ax in axes[len(samples):]:
        ax.axis('off')
    
    plt.suptitle("Random Dataset Samples - Look for patterns in 'noise'", fontsize=14)
    plt.tight_layout()
    plt.savefig("dataset_samples.png", dpi=150, bbox_inches='tight')
    print("   📸 Saved grid to 'dataset_samples.png'")
    plt.show()


def create_labeling_subset(image_paths: list, num_images: int = 100, output_dir: str = "labeling_subset"):
    """
    Copy a random subset of images to a separate folder for manual labeling.
    
    Why 100 images?
    - Enough to get statistically meaningful evaluation metrics
    - Small enough to label in 30-60 minutes
    - Industry standard for quick validation sets
    
    Args:
        image_paths: All image paths
        num_images: How many to sample for labeling
        output_dir: Where to copy the images
    """
    import shutil
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Randomly sample
    samples = random.sample(image_paths, min(num_images, len(image_paths)))
    
    print(f"\n📋 Creating labeling subset of {len(samples)} images in '{output_dir}/'")
    
    for i, src_path in enumerate(samples):
        # Keep original filename but add index prefix for easy sorting
        filename = f"{i:03d}_{os.path.basename(src_path)}"
        dst_path = os.path.join(output_dir, filename)
        shutil.copy2(src_path, dst_path)
    
    print(f"   ✅ Copied {len(samples)} images")
    print(f"\n   Next: Create 'labels.csv' in this folder with your manual labels.")
    print(f"   See instructions in the README for labeling format.")
    
    # Create a template CSV for labeling
    csv_template = os.path.join(output_dir, "labels_template.csv")
    with open(csv_template, 'w') as f:
        f.write("filename,full_body,face_visible,is_adult,not_advertisement,keep\n")
        for i, src_path in enumerate(samples):
            filename = f"{i:03d}_{os.path.basename(src_path)}"
            # Pre-fill with empty values for manual labeling
            f.write(f"{filename},,,,,\n")
    
    print(f"   📝 Created '{csv_template}' - fill this out during labeling!")


def plot_size_distribution(size_analysis: dict):
    """
    Create visualizations of image size distributions.
    
    Why visualize?
    - Quickly spot outliers (tiny or huge images)
    - Understand if you need to resize/normalize
    - Check for bimodal distributions (multiple image sources)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Width distribution
    axes[0, 0].hist(size_analysis['widths'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Image Width Distribution')
    axes[0, 0].axvline(np.median(size_analysis['widths']), color='r', linestyle='--', label='Median')
    axes[0, 0].legend()
    
    # Height distribution
    axes[0, 1].hist(size_analysis['heights'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Image Height Distribution')
    axes[0, 1].axvline(np.median(size_analysis['heights']), color='r', linestyle='--', label='Median')
    axes[0, 1].legend()
    
    # Aspect ratio distribution
    axes[1, 0].hist(size_analysis['aspect_ratios'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Aspect Ratio (width/height)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Aspect Ratio Distribution')
    axes[1, 0].axvline(1.0, color='r', linestyle='--', label='Square (1:1)')
    axes[1, 0].legend()
    
    # File size distribution
    axes[1, 1].hist(size_analysis['file_sizes_kb'], bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('File Size (KB)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('File Size Distribution')
    axes[1, 1].axvline(np.median(size_analysis['file_sizes_kb']), color='r', linestyle='--', label='Median')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig("size_distributions.png", dpi=150, bbox_inches='tight')
    print("\n📊 Saved size distribution plots to 'size_distributions.png'")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("🔍 DATASET EXPLORATION FOR COMPUTER VISION")
    print("="*60)
    
    # Step 1: Find all images
    print(f"\n📂 Scanning: {DATASET_PATH}")
    image_paths = get_all_image_paths(DATASET_PATH)
    
    if not image_paths:
        print("❌ No images found! Check your DATASET_PATH.")
        exit(1)
    
    print(f"✅ Found {len(image_paths)} images")
    
    # Step 2: Analyze sizes
    size_analysis = analyze_image_sizes(image_paths)
    
    # Step 3: Print statistics
    print_statistics(image_paths, size_analysis)
    
    # Step 4: Plot distributions
    plot_size_distribution(size_analysis)
    
    # Step 5: Show random samples
    display_random_samples(image_paths, NUM_SAMPLES_TO_VIEW)
    
    # Step 6: Create labeling subset
    create_labeling_subset(image_paths, num_images=100)
    
    print("\n" + "="*60)
    print("✅ EXPLORATION COMPLETE!")
    print("="*60)
    print("""
NEXT STEPS:
1. Review 'dataset_samples.png' to understand the noise
2. Review 'size_distributions.png' for size patterns
3. Go to 'labeling_subset/' folder
4. Open each image and fill out 'labels_template.csv'
5. Use your labels to evaluate your pipeline later
    """)
