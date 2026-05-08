"""
Test Data Preparation Helper Script
================================================================================
Helps organize test images by cattle breed class for evaluation.

Usage:
    python prepare_test_data.py --input-dir /path/to/images --output-dir ./test_data

================================================================================
"""

import os
import shutil
import argparse
import json
from pathlib import Path
from collections import defaultdict


def organize_test_data(input_dir, output_dir, class_names):
    """
    Organize test images by class.
    
    Can handle:
    1. Images named with class prefix: Gir_001.jpg, Gir_002.jpg, etc.
    2. Images in subdirectories: input_dir/Gir/img1.jpg, etc.
    3. Manual specification via CSV: image_name,class_name
    
    Args:
        input_dir: Source directory with test images
        output_dir: Destination directory for organized images
        class_names: List of class names to organize by
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each class
    for class_name in class_names:
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
    
    organized_count = defaultdict(int)
    unorganized_images = []
    
    # Process images
    for img_file in input_path.glob('*'):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            continue
        
        # Try to match class from filename
        filename = img_file.stem.lower()
        matched_class = None
        
        for class_name in class_names:
            if class_name.lower() in filename:
                matched_class = class_name
                break
        
        if matched_class:
            # Copy to class directory
            dest_path = output_path / matched_class / img_file.name
            shutil.copy2(img_file, dest_path)
            organized_count[matched_class] += 1
            print(f"✅ {img_file.name} → {matched_class}/")
        else:
            unorganized_images.append(img_file.name)
    
    # Summary
    print("\n" + "="*60)
    print("📊 ORGANIZATION SUMMARY")
    print("="*60)
    total_organized = sum(organized_count.values())
    print(f"\nTotal images organized: {total_organized}")
    
    if organized_count:
        print("\nImages per class:")
        for class_name in sorted(organized_count.keys()):
            count = organized_count[class_name]
            print(f"  • {class_name}: {count}")
    
    if unorganized_images:
        print(f"\n⚠️  Could not organize {len(unorganized_images)} images:")
        for img in unorganized_images[:10]:
            print(f"  • {img}")
        if len(unorganized_images) > 10:
            print(f"  ... and {len(unorganized_images) - 10} more")
    
    print(f"\n✅ Organized data saved to: {output_path}")


def create_test_subset(class_dir, output_dir, images_per_class=5):
    """
    Create a small test subset from organized data.
    Useful for quick testing before running full evaluation.
    """
    class_path = Path(class_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for class_subdir in class_path.iterdir():
        if not class_subdir.is_dir():
            continue
        
        dest_dir = output_path / class_subdir.name
        dest_dir.mkdir(exist_ok=True)
        
        # Copy first N images from each class
        images = list(class_subdir.glob('*'))[:images_per_class]
        for img in images:
            shutil.copy2(img, dest_dir / img.name)
        
        print(f"✅ {class_subdir.name}: copied {len(images)} images")
    
    print(f"\n✅ Test subset created: {output_path}")


def load_metadata(metadata_path):
    """Load class names from metadata."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata['class_names']


def main():
    parser = argparse.ArgumentParser(
        description="Organize test images by cattle breed class"
    )
    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Directory containing test images'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./test_data',
        help='Output directory for organized data (default: ./test_data)'
    )
    parser.add_argument(
        '--metadata-path', '-m',
        default='./metadata.json',
        help='Path to metadata.json with class names'
    )
    parser.add_argument(
        '--subset',
        action='store_true',
        help='Create a small test subset (5 images per class)'
    )
    parser.add_argument(
        '--subset-size',
        type=int,
        default=5,
        help='Number of images per class for subset'
    )
    
    args = parser.parse_args()
    
    print("🚀 Test Data Preparation Tool")
    print("="*60)
    
    # Load class names
    print("\n📂 Loading class names...")
    try:
        class_names = load_metadata(args.metadata_path)
        print(f"✅ Found {len(class_names)} classes")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        print("   Please provide correct path with --metadata-path")
        return
    
    # Organize images
    print("\n📊 Organizing images...")
    organize_test_data(args.input_dir, args.output_dir, class_names)
    
    # Create subset if requested
    if args.subset:
        print("\n🔍 Creating test subset...")
        subset_dir = f"{args.output_dir}_subset"
        create_test_subset(args.output_dir, subset_dir, args.subset_size)


if __name__ == "__main__":
    main()
