#!/usr/bin/env python3
"""
Dataset Validator Script
Checks for grayscale images and other potential issues in image datasets.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

class DatasetValidator:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.results = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'grayscale_images': 0,
            'rgb_images': 0,
            'other_formats': 0,
            'missing_files': 0,
            'corrupted_files': 0,
            'invalid_paths': [],
            'grayscale_paths': [],
            'corrupted_paths': [],
            'missing_paths': [],
            'class_distribution': {},
            'image_sizes': [],
            'issues_found': []
        }
    
    def load_csv(self) -> bool:
        """Load and validate CSV file."""
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # Check if required columns exist
            if 'path' not in self.df.columns or 'label' not in self.df.columns:
                # Try to infer column names
                if len(self.df.columns) >= 2:
                    self.df.columns = ['path', 'label'] + list(self.df.columns[2:])
                else:
                    self.results['issues_found'].append("CSV must have at least 2 columns: path and label")
                    return False
            
            self.results['total_images'] = len(self.df)
            print(f"✓ Loaded CSV with {self.results['total_images']} images")
            return True
            
        except Exception as e:
            self.results['issues_found'].append(f"Error loading CSV: {str(e)}")
            return False
    
    def validate_image(self, image_path: str) -> Dict:
        """Validate a single image and return its properties."""
        result = {
            'valid': False,
            'exists': False,
            'readable': False,
            'format': None,
            'channels': 0,
            'size': None,
            'error': None
        }
        
        # Check if file exists
        if not os.path.exists(image_path):
            result['error'] = 'File does not exist'
            return result
        
        result['exists'] = True
        
        # Try to open and analyze image
        try:
            with Image.open(image_path) as img:
                result['readable'] = True
                result['size'] = img.size
                result['format'] = img.format
                
                # Convert to array to check channels
                img_array = np.array(img)
                
                if len(img_array.shape) == 2:
                    # Grayscale image
                    result['channels'] = 1
                    result['format'] = 'grayscale'
                elif len(img_array.shape) == 3:
                    # Color image
                    result['channels'] = img_array.shape[2]
                    if result['channels'] == 3:
                        result['format'] = 'RGB'
                    elif result['channels'] == 4:
                        result['format'] = 'RGBA'
                    else:
                        result['format'] = f'color_{result["channels"]}ch'
                else:
                    result['format'] = 'unknown'
                    result['error'] = f'Unexpected image shape: {img_array.shape}'
                    return result
                
                result['valid'] = True
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def analyze_dataset(self) -> Dict:
        """Analyze the entire dataset for issues."""
        if self.df is None:
            return self.results
        
        print("🔍 Analyzing dataset...")
        
        # Analyze each image
        for idx, row in self.df.iterrows():
            image_path = row['path']
            label = row['label']
            
            # Validate image
            validation = self.validate_image(image_path)
            
            if not validation['exists']:
                self.results['missing_files'] += 1
                self.results['missing_paths'].append(image_path)
            elif not validation['readable']:
                self.results['corrupted_files'] += 1
                self.results['corrupted_paths'].append(image_path)
            elif validation['valid']:
                self.results['valid_images'] += 1
                
                # Count by format
                if validation['channels'] == 1:
                    self.results['grayscale_images'] += 1
                    self.results['grayscale_paths'].append(image_path)
                elif validation['channels'] == 3:
                    self.results['rgb_images'] += 1
                else:
                    self.results['other_formats'] += 1
                
                # Store image size
                if validation['size']:
                    self.results['image_sizes'].append(validation['size'])
            else:
                self.results['invalid_images'] += 1
                self.results['invalid_paths'].append(image_path)
        
        # Calculate class distribution
        if 'label' in self.df.columns:
            self.results['class_distribution'] = self.df['label'].value_counts().to_dict()
        
        # Check for potential issues
        self._check_for_issues()
        
        return self.results
    
    def _check_for_issues(self):
        """Check for specific issues that could cause problems."""
        issues = []
        
        # Check for grayscale images (potential issue for RGB-based processing)
        if self.results['grayscale_images'] > 0:
            issues.append(f"Found {self.results['grayscale_images']} grayscale images - these may cause inhomogeneous array errors")
        
        # Check for mixed formats
        if self.results['grayscale_images'] > 0 and self.results['rgb_images'] > 0:
            issues.append("Mixed grayscale and RGB images detected - this will cause inhomogeneous array errors")
        
        # Check for missing files
        if self.results['missing_files'] > 0:
            issues.append(f"Found {self.results['missing_files']} missing files")
        
        # Check for corrupted files
        if self.results['corrupted_files'] > 0:
            issues.append(f"Found {self.results['corrupted_files']} corrupted files")
        
        # Check class balance
        if self.results['class_distribution']:
            class_counts = list(self.results['class_distribution'].values())
            max_count = max(class_counts)
            min_count = min(class_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 2:
                issues.append(f"Class imbalance detected: ratio {imbalance_ratio:.2f} (max/min)")
        
        self.results['issues_found'].extend(issues)
    
    def generate_report(self) -> str:
        """Generate a detailed report."""
        report = []
        report.append("=" * 60)
        report.append("DATASET VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"CSV File: {self.csv_path}")
        report.append(f"Total Images: {self.results['total_images']}")
        report.append("")
        
        # Summary
        report.append("📊 SUMMARY:")
        report.append(f"  ✓ Valid Images: {self.results['valid_images']}")
        report.append(f"  ❌ Invalid Images: {self.results['invalid_images']}")
        report.append(f"  📁 Missing Files: {self.results['missing_files']}")
        report.append(f"  💥 Corrupted Files: {self.results['corrupted_files']}")
        report.append("")
        
        # Image formats
        report.append("🖼️  IMAGE FORMATS:")
        report.append(f"  RGB Images: {self.results['rgb_images']}")
        report.append(f"  Grayscale Images: {self.results['grayscale_images']}")
        report.append(f"  Other Formats: {self.results['other_formats']}")
        report.append("")
        
        # Class distribution
        if self.results['class_distribution']:
            report.append("🏷️  CLASS DISTRIBUTION:")
            for class_name, count in self.results['class_distribution'].items():
                percentage = (count / self.results['total_images']) * 100
                report.append(f"  {class_name}: {count} ({percentage:.1f}%)")
            report.append("")
        
        # Image sizes
        if self.results['image_sizes']:
            widths = [size[0] for size in self.results['image_sizes']]
            heights = [size[1] for size in self.results['image_sizes']]
            report.append("📏 IMAGE SIZES:")
            report.append(f"  Width Range: {min(widths)} - {max(widths)}")
            report.append(f"  Height Range: {min(heights)} - {max(heights)}")
            report.append(f"  Average Size: {np.mean(widths):.0f} x {np.mean(heights):.0f}")
            report.append("")
        
        # Issues
        if self.results['issues_found']:
            report.append("⚠️  ISSUES FOUND:")
            for issue in self.results['issues_found']:
                report.append(f"  • {issue}")
            report.append("")
        else:
            report.append("✅ No issues found!")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        if self.results['grayscale_images'] > 0:
            report.append("  • Convert grayscale images to RGB to avoid inhomogeneous array errors")
            report.append("  • Update your processing code to handle mixed formats")
        
        if self.results['missing_files'] > 0:
            report.append("  • Check file paths and ensure all images exist")
        
        if self.results['corrupted_files'] > 0:
            report.append("  • Re-download or fix corrupted image files")
        
        if not self.results['issues_found']:
            report.append("  • Dataset looks good! No immediate issues detected.")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_detailed_report(self, output_path: str):
        """Save detailed results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Detailed report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Validate image dataset for potential issues')
    parser.add_argument('csv_path', help='Path to the CSV file containing image paths and labels')
    parser.add_argument('--output', '-o', help='Output file for detailed JSON report')
    parser.add_argument('--fix-grayscale', action='store_true', help='Generate a script to convert grayscale images to RGB')
    
    args = parser.parse_args()
    
    # Validate CSV file exists
    if not os.path.exists(args.csv_path):
        print(f"❌ Error: CSV file '{args.csv_path}' not found")
        return 1
    
    # Create validator and run analysis
    validator = DatasetValidator(args.csv_path)
    
    if not validator.load_csv():
        print("❌ Failed to load CSV file")
        return 1
    
    results = validator.analyze_dataset()
    
    # Print report
    print(validator.generate_report())
    
    # Save detailed report if requested
    if args.output:
        validator.save_detailed_report(args.output)
    
    # Generate fix script if requested
    if args.fix_grayscale and results['grayscale_images'] > 0:
        generate_fix_script(results['grayscale_paths'])
    
    # Return appropriate exit code
    return 0 if not results['issues_found'] else 1

def generate_fix_script(grayscale_paths: List[str]):
    """Generate a script to convert grayscale images to RGB."""
    script_content = '''#!/usr/bin/env python3
"""
Script to convert grayscale images to RGB format.
Generated by dataset_validator.py
"""

import os
from PIL import Image
import argparse

def convert_to_rgb(image_path):
    """Convert a grayscale image to RGB."""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                rgb_img = img.convert('RGB')
                rgb_img.save(image_path, 'JPEG', quality=95)
                print(f"✓ Converted: {image_path}")
                return True
    except Exception as e:
        print(f"❌ Error converting {image_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert grayscale images to RGB')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be converted without making changes')
    args = parser.parse_args()
    
    grayscale_paths = [
'''
    
    for path in grayscale_paths:
        script_content += f"        '{path}',\n"
    
    script_content += '''    ]
    
    print(f"Found {len(grayscale_paths)} grayscale images to convert")
    
    if args.dry_run:
        print("DRY RUN - No changes will be made:")
        for path in grayscale_paths:
            print(f"  Would convert: {path}")
    else:
        print("Converting grayscale images to RGB...")
        converted = 0
        for path in grayscale_paths:
            if convert_to_rgb(path):
                converted += 1
        
        print(f"✓ Converted {converted}/{len(grayscale_paths)} images")

if __name__ == "__main__":
    main()
'''
    
    with open('fix_grayscale_images.py', 'w') as f:
        f.write(script_content)
    
    print("🔧 Generated fix script: fix_grayscale_images.py")
    print("   Run: python fix_grayscale_images.py --dry-run  # to see what would be converted")
    print("   Run: python fix_grayscale_images.py            # to convert the images")

if __name__ == "__main__":
    exit(main()) 