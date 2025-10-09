"""
Training Image Generator for YOLO Object Detection

This module provides functionality to generate synthetic training images for YOLO object detection
by placing objects on various backgrounds with augmentations like rotation, noise, and exposure adjustments.

The main class `TrainingImageGenerator` supports:
- Multiple background modes (blank, specific images, random selection)
- Noise generation for background augmentation  
- Balanced class sampling for training data
- Support for both bounding box and polygon label formats
- Rotation and exposure augmentations
- Grid-based object placement with collision avoidance

Example:
    Basic usage:
    ```python
    generator = TrainingImageGenerator(
        images_dir="./train/images",
        labels_dir="./train/labels",
        background_mode="random",
        use_noisy_background=True
    )
    
    # Generate 10 training images
    generated_files = generator.generate_multiple_images(n_images=10)
    ```

Date:
    2025
"""

import os
import cv2
import numpy as np
import random
import math
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

class TrainingImageGenerator:
    """
    A comprehensive tool for generating synthetic training images for YOLO object detection.
    
    This class creates training images by placing objects from existing datasets onto various
    backgrounds with augmentations to improve model robustness. It supports balanced class
    sampling, multiple background modes, and various augmentation techniques.
    
    Attributes:
        images_dir (Path): Directory containing training images
        labels_dir (Path): Directory containing YOLO format label files  
        backgrounds_dir (Path): Directory containing background images
        canvas_width (int): Width of generated images (default: 3076)
        canvas_height (int): Height of generated images (default: 1852)
        cell_size (int): Size of grid cells for object placement (default: 250)
        background_mode (str): Background selection mode
        use_noisy_background (bool): Whether to add noise to backgrounds
        noise_intensity (float): Intensity of background noise (0.0-1.0)
        
    Background Modes:
        - "blank": Pure white background
        - "random": Random choice between blank and all available backgrounds
        - "random_backgrounds": Random choice from available backgrounds only
        - Specific name: Use a specific background image by name
        
    Example:
        ```python
        # Initialize with balanced sampling and random backgrounds
        generator = TrainingImageGenerator(
            images_dir="./train/images",
            labels_dir="./train/labels", 
            background_mode="random",
            use_noisy_background=True,
            noise_intensity=0.5
        )
        
        # Generate training images
        files = generator.generate_multiple_images(n_images=100)
        ```
    """
    def __init__(self, images_dir="./train/images", labels_dir="./train/labels", use_noisy_background=False, noise_intensity=0.3, 
                 background_mode="blank", backgrounds_dir="../echteHintergründe", random_noise_intensity=False):
        """
        Initialize the Training Image Generator.
        
        Args:
            images_dir (str): Path to directory containing training images
            labels_dir (str): Path to directory containing YOLO label files  
            use_noisy_background (bool): Enable background noise generation
            noise_intensity (float): Noise intensity level (0.0-1.0, ignored if random_noise_intensity=True)
            background_mode (str): Background selection mode
            backgrounds_dir (str): Path to background images directory
            random_noise_intensity (bool): If True, randomly select noise intensity for each image (0.0-1.0)
            
        Raises:
            ValueError: If no training data is found in specified directories
            
        Note:
            Paths are automatically adjusted based on execution context (parent vs Screws_Png directory)
        """
        # Image Directories based on current file path and execution context
        
        # Determine if we're being run from parent directory or Screws_Png directory
        current_path = Path(__file__).parent
        
        # Check if we're in the Screws_Png directory or parent directory
        if current_path.name == "Screws_Png":
            # Running from Screws_Png directory - use relative paths as-is
            self.images_dir = Path(images_dir)
            self.labels_dir = Path(labels_dir)
            self.backgrounds_dir = Path(backgrounds_dir)
        else:
            # Running from parent directory - adjust paths to point to Screws_Png subdirectory
            screws_png_path = current_path / "Screws_Png"
            self.images_dir = screws_png_path / images_dir
            self.labels_dir = screws_png_path / labels_dir
            self.backgrounds_dir = current_path / "echteHintergründe"  # echteHintergründe is in parent dir
        self.canvas_width = 3076
        self.canvas_height = 1852
        self.cell_size = 250
        
        # Background selection settings
        self.background_mode = background_mode  # "blank", any background filename, "random", "random_backgrounds"
        self.background_images = {}
        
        # Noise background settings
        self.use_noisy_background = use_noisy_background
        self.noise_intensity = noise_intensity  # 0.0 to 1.0, controls how noisy the background is
        self.random_noise_intensity = random_noise_intensity  # Whether to randomize noise intensity per image
        
        # Calculate grid dimensions
        self.grid_cols = self.canvas_width // self.cell_size
        self.grid_rows = self.canvas_height // self.cell_size
        
        print(f"Canvas size: {self.canvas_width}x{self.canvas_height}")
        print(f"Grid size: {self.grid_cols}x{self.grid_rows} cells")
        print(f"Cell size: {self.cell_size}x{self.cell_size}")
        print(f"Background mode: {self.background_mode}")
        
        # Load background images
        self.load_background_images()
        
        # Load available PNG files and their labels
        self.load_training_data()
    
    def load_background_images(self):
        """
        Load background images from the backgrounds directory.
        
        Loads all PNG files from the backgrounds directory and handles Unicode paths
        that cv2 might have trouble with. Validates the selected background mode.
        
        Note:
            Images are loaded with proper Unicode path handling using numpy.fromfile()
            and cv2.imdecode() to avoid path encoding issues.
            
        Side Effects:
            - Populates self.background_images dictionary
            - Validates and potentially resets self.background_mode
            - Prints loading status and available backgrounds
        """
        """Load background images from echteHintergründe directory"""
        self.background_images = {}
        
        if not self.backgrounds_dir.exists():
            print(f"⚠️ Background directory not found: {self.backgrounds_dir}")
            print(f"   Will use blank backgrounds only")
            return
        
        # Load all PNG files from backgrounds directory
        png_files = list(self.backgrounds_dir.glob("*.png"))
        
        for bg_path in png_files:
            bg_name = bg_path.stem  # Get filename without extension
            if bg_path.exists():
                try:
                    # Load image with proper Unicode path handling
                    # Use numpy to handle Unicode paths that cv2 can't handle
                    image_array = np.fromfile(str(bg_path), dtype=np.uint8)
                    bg_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
                    
                    if bg_image is not None:
                        # Convert BGR to RGB if it's a 3-channel image
                        if len(bg_image.shape) == 3 and bg_image.shape[2] == 3:
                            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
                        elif len(bg_image.shape) == 3 and bg_image.shape[2] == 4:
                            # For RGBA images, convert BGRA to RGBA
                            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGRA2RGBA)
                        
                        # Resize to canvas size
                        bg_image = cv2.resize(bg_image, (self.canvas_width, self.canvas_height), 
                                            interpolation=cv2.INTER_AREA)
                        
                        self.background_images[bg_name] = bg_image
                        print(f"   ✅ Loaded background: {bg_name}.png ({bg_image.shape})")
                    else:
                        print(f"   ❌ Could not decode: {bg_name}.png")
                except Exception as e:
                    print(f"   ❌ Error loading {bg_name}.png: {e}")
            else:
                print(f"   ⚠️ Background not found: {bg_name}.png")
        
        print(f"📁 Loaded {len(self.background_images)} background images")
        
        # Validate background mode
        # Valid modes include: blank, any loaded background name, random, random_backgrounds
        specific_background_modes = list(self.background_images.keys())
        valid_modes = ["blank", "random", "random_backgrounds"] + specific_background_modes
        
        if self.background_mode not in valid_modes:
            print(f"⚠️ Invalid background mode '{self.background_mode}', defaulting to 'blank'")
            self.background_mode = "blank"
        
        # Check if specific background is available
        if self.background_mode in specific_background_modes and self.background_mode not in self.background_images:
            print(f"⚠️ Background '{self.background_mode}' not available, defaulting to 'blank'")
            self.background_mode = "blank"
    
    def load_training_data(self):
        """
        Load all PNG files and their corresponding YOLO format labels.
        
        Searches for PNG files in the images directory and matches them with
        corresponding label files. Handles both positive examples (with objects)
        and negative examples (e.g. screws that are not part of the assembly).
        
        The method performs color conversion from BGR to RGB for proper display
        and categorizes examples as positive or negative based on label presence.
        
        Raises:
            ValueError: If no training data is found in the specified directories
            
        Side Effects:
            - Populates self.training_data list with image and label information
            - Calls analyze_class_distribution() for balanced sampling preparation
            - Prints loading statistics and file counts
        """
        """Load all PNG files and their corresponding labels"""
        self.training_data = []
        
        # Find all PNG files
        png_files = list(self.images_dir.glob("*.png"))
        
        print(f"Found {len(png_files)} PNG files in {self.images_dir}")
        
        for png_file in png_files:
            # Find corresponding label file
            label_file = self.labels_dir / f"{png_file.stem}.txt"
            
            if label_file.exists():
                print(f"\nProcessing: {png_file.name}")
                
                # Load image with proper color handling
                image = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    # Convert BGR to RGB if it's a 3-channel image (fixes color issue)
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    elif len(image.shape) == 3 and image.shape[2] == 4:
                        # For RGBA images, convert BGRA to RGBA
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                    
                    # Apply exposure adjustment for Standing-Nut and Standing-Screw files
                    # filename_lower = png_file.name.lower()
                    # if 'standing-nut' in filename_lower or 'standing-screw' in filename_lower:
                    #     image = self.adjust_exposure(image, exposure_increase=0.5)
                    #     print(f"  📸 Applied exposure adjustment for {png_file.name}")
                    
                    # Load labels
                    labels = self.load_yolo_labels(label_file)
                    
                    # Add to training data regardless of whether labels exist
                    # Empty labels indicate negative examples (background-only)
                    self.training_data.append({
                        'image': image,
                        'labels': labels,
                        'filename': png_file.name,
                        'type': 'positive' if labels else 'negative'
                    })
                    
                    if labels:
                        print(f"  ✅ Added {len(labels)} labels (positive example)")
                    else:
                        print(f"  🔲 Added as negative example (empty labels)")
                else:
                    print(f"  Could not load image")
            else:
                print(f"Missing label file for {png_file.name}")
        
        # Count positive and negative examples
        positive_count = sum(1 for data in self.training_data if data['type'] == 'positive')
        negative_count = sum(1 for data in self.training_data if data['type'] == 'negative')
        
        print(f"\n📊 Summary:")
        print(f"   Positive examples (with objects): {positive_count}")
        print(f"   Negative examples (backgrounds): {negative_count}")
        print(f"   Total training images: {len(self.training_data)}")
        
        if len(self.training_data) == 0:
            print("No training data found!")
            print("Please check:")
            print(f"   - PNG files are in: {self.images_dir}")
            print(f"   - Label files are in: {self.labels_dir}")
            print(f"   - Label files have same name as PNG files (but .txt extension)")
            raise ValueError("No training data found! Please check your paths.")
        
        # Analyze class distribution for balanced sampling
        self.analyze_class_distribution()
    
    def analyze_class_distribution(self):
        """
        Analyze class distribution and prepare for balanced sampling.
        
        Examines the training data to count class occurrences and identify
        underrepresented classes. Creates mappings for efficient balanced sampling.
        
        Side Effects:
            - Populates self.class_distribution with class counts
            - Creates self.class_to_files mapping for sampling
            - Identifies self.underrepresented_classes for prioritization
            - Prints detailed distribution analysis and balance metrics
            
        Note:
            Classes with less than 50% of the most common class count are
            considered underrepresented and will be prioritized in balanced sampling.
        """
        """Analyze class distribution and prepare for balanced sampling"""
        self.class_distribution = {}
        self.class_to_files = {}  # Maps class_id to list of file indices
        
        # Count positive and negative examples
        positive_count = sum(1 for data in self.training_data if data['type'] == 'positive')
        negative_count = sum(1 for data in self.training_data if data['type'] == 'negative')
        
        # Count classes and map to files (only for positive examples)
        for idx, data in enumerate(self.training_data):
            if data['type'] == 'positive':  # Only analyze positive examples for class distribution
                for label in data['labels']:
                    class_id = label['class_id']
                    
                    # Count occurrences
                    self.class_distribution[class_id] = self.class_distribution.get(class_id, 0) + 1
                    
                    # Map class to files containing this class
                    if class_id not in self.class_to_files:
                        self.class_to_files[class_id] = []
                    if idx not in self.class_to_files[class_id]:
                        self.class_to_files[class_id].append(idx)
        
        # Find underrepresented classes
        max_count = max(self.class_distribution.values()) if self.class_distribution else 0
        min_count = min(self.class_distribution.values()) if self.class_distribution else 0
        
        self.underrepresented_classes = []
        for class_id, count in self.class_distribution.items():
            if count < max_count * 0.5:  # Less than 50% of most common class
                self.underrepresented_classes.append(class_id)
        
        print(f"\n🎯 CLASS DISTRIBUTION ANALYSIS")
        print("=" * 50)
        print(f"Total classes found: {len(self.class_distribution)}")
        print(f"Negative examples available: {negative_count}")
        print(f"Class distribution:")
        
        # Class names for display
        class_names = [
            "M4-12mm", "M4-16mm", "M4-20mm", "M4-8mm", "M4-Nut", "M4-Washer",
            "M6-12mm", "M6-16mm", "M6-20mm", "M6-8mm", "M6-Nut", "M6-Washer",
            "Standing-Nut", "Standing-Screw"
        ]
        
        for class_id in sorted(self.class_distribution.keys()):
            count = self.class_distribution[class_id]
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            files_with_class = len(self.class_to_files[class_id])
            status = "⚠️ UNDERREPRESENTED" if class_id in self.underrepresented_classes else "✅"
            print(f"   Class {class_id:2d} ({class_name:15s}): {count:3d} objects in {files_with_class:2d} files {status}")
        
        print(f"\nBalance analysis:")
        print(f"   Most common class: {max_count} objects")
        print(f"   Least common class: {min_count} objects")
        print(f"   Underrepresented classes: {len(self.underrepresented_classes)} ({self.underrepresented_classes})")
        print(f"   🔲 Negative examples: {negative_count} images")
        
        if self.underrepresented_classes:
            print(f"   ⚡ Balanced sampling will prioritize underrepresented classes")
        else:
            print(f"   ✅ Classes are well balanced")
        
        print("=" * 50)
    
    def load_yolo_labels(self, label_file):
        """
        Load YOLO format labels from file - preserves both box and polygon formats.
        
        Parses YOLO format label files supporting both standard bounding box format
        (5 values: class x_center y_center width height) and polygon format 
        (variable length: class x1 y1 x2 y2 ... xn yn).
        
        Args:
            label_file (Path): Path to the YOLO format label file
            
        Returns:
            list[dict]: List of label dictionaries with format:
                {
                    'type': 'box' or 'polygon',
                    'class_id': int,
                    'data': list of float coordinates
                }
                
        Note:
            - Box format: [x_center, y_center, width, height] (normalized 0-1)
            - Polygon format: [x1, y1, x2, y2, ..., xn, yn] (normalized 0-1)
            - Invalid lines are skipped with warning messages
        """
        """Load YOLO format labels from file - preserves both box and polygon formats"""
        labels = []
        try:
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f.readlines(), 1):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    
                    if len(parts) == 5:
                        # Standard YOLO box format: class x_center y_center width height
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append({
                            'type': 'box',
                            'class_id': class_id,
                            'data': [x_center, y_center, width, height]
                        })
                        print(f"  Line {line_num}: Box format - Class {class_id}")
                        
                    elif len(parts) >= 9 and len(parts) % 2 == 1:
                        # Polygon format: class x1 y1 x2 y2 x3 y3 x4 y4 [x5 y5 ...]
                        # Keep original polygon points
                        num_points = (len(parts) - 1) // 2
                        points = [float(parts[i]) for i in range(1, len(parts))]
                        
                        labels.append({
                            'type': 'polygon',
                            'class_id': class_id,
                            'data': points
                        })
                        print(f"  Line {line_num}: Polygon format - Class {class_id}, {num_points} points")
                        
                    else:
                        print(f"  Line {line_num}: Unknown format with {len(parts)} parts - skipping")
                        
        except Exception as e:
            print(f"Error loading labels from {label_file}: {e}")
        
        return labels
    
    def adjust_exposure(self, image, exposure_increase=0.3):
        """
        Adjust exposure/brightness of an image, preserving alpha channel if present.
        Args:
            image (numpy.ndarray): Input image
            exposure_increase (float): Amount to increase exposure (0.0-1.0)

        Returns:
            numpy.ndarray: Exposure-adjusted image
        """
        adjusted_image = image.copy()
        
        if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            # Adjust only RGB channels, preserve alpha
            rgb_channels = adjusted_image[:, :, :3].astype(np.float32)
            alpha_channel = adjusted_image[:, :, 3]
            
            # Apply gamma correction for exposure adjustment
            # gamma < 1 makes image brighter, gamma > 1 makes it darker
            gamma = 1.0 - exposure_increase  # exposure_increase of 0.3 gives gamma of 0.7
            gamma = max(0.1, gamma)  # Ensure gamma doesn't go too low
            
            # Normalize to 0-1, apply gamma, then scale back to 0-255
            rgb_normalized = rgb_channels / 255.0
            rgb_corrected = np.power(rgb_normalized, gamma)
            rgb_adjusted = (rgb_corrected * 255.0).astype(np.uint8)
            
            # Recombine with alpha channel
            adjusted_image[:, :, :3] = rgb_adjusted
            adjusted_image[:, :, 3] = alpha_channel
            
        elif len(image.shape) == 3:  # RGB
            # Adjust all RGB channels
            rgb_channels = adjusted_image.astype(np.float32)
            
            # Apply gamma correction
            gamma = 1.0 - exposure_increase
            gamma = max(0.1, gamma)
            
            rgb_normalized = rgb_channels / 255.0
            rgb_corrected = np.power(rgb_normalized, gamma)
            adjusted_image = (rgb_corrected * 255.0).astype(np.uint8)
            
        else:  # Grayscale
            # Apply gamma correction to grayscale
            gray_normalized = adjusted_image.astype(np.float32) / 255.0
            gamma = 1.0 - exposure_increase
            gamma = max(0.1, gamma)
            gray_corrected = np.power(gray_normalized, gamma)
            adjusted_image = (gray_corrected * 255.0).astype(np.uint8)
        
        return adjusted_image
    
    def create_shadow(self, image, shadow_offset_x=None, shadow_offset_y=None, shadow_intensity=0.3, return_shadow_layer=False):
        """
        Create a realistic stretched shadow that appears cast on a surface.
        
        The shadow is rooted to the object's base and stretched away from the light source,
        extending beyond the original image boundaries for a natural ground-cast shadow effect.
        
        Args:
            image (numpy.ndarray): Input RGBA image with transparent background
            shadow_offset_x (int, optional): Horizontal shadow direction. If None, random value is used.
            shadow_offset_y (int, optional): Vertical shadow direction. If None, random value is used.
            shadow_intensity (float): Shadow darkness (0.0-1.0, where 1.0 is darkest)
            
        Returns:
            numpy.ndarray: Image with realistic stretched shadow, possibly expanded canvas
        """
        if len(image.shape) != 3 or image.shape[2] != 4:
            # No alpha channel, return original image
            return image
        
        # Random shadow parameters if not provided
        if shadow_offset_x is None:
            shadow_offset_x = random.randint(-30, 30)  # Shadow direction
        if shadow_offset_y is None:
            shadow_offset_y = random.randint(20, 60)   # Shadow length (mostly downward)
        
        # 10% chance for upward shadow (different lighting)
        if random.random() < 0.1:
            shadow_offset_y = -abs(shadow_offset_y)
        
        h, w = image.shape[:2]
        
        # Find object bounds for shadow rooting
        alpha_channel = image[:, :, 3]
        object_coords = np.where(alpha_channel > 0)
        if len(object_coords[0]) == 0:
            return image  # No object pixels
        
        # Find object's bottom center point (shadow root)
        bottom_y = np.max(object_coords[0])  # Lowest point of object
        center_x = int(np.mean(object_coords[1]))  # Center x of object
        
        # Calculate expanded canvas size to accommodate stretched shadow
        shadow_extent_x = abs(shadow_offset_x) + 20
        shadow_extent_y = abs(shadow_offset_y) + 20
        
        new_w = w + shadow_extent_x * 2
        new_h = h + shadow_extent_y * 2
        
        # Create expanded canvas
        expanded_canvas = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        
        # Place original image in center of expanded canvas
        orig_offset_x = shadow_extent_x
        orig_offset_y = shadow_extent_y
        expanded_canvas[orig_offset_y:orig_offset_y+h, orig_offset_x:orig_offset_x+w] = image
        
        # Create shadow using skew transformation
        # Transform the object's alpha channel to create stretched shadow
        
        # Calculate shadow anchor point in expanded canvas
        shadow_root_x = orig_offset_x + center_x
        shadow_root_y = orig_offset_y + bottom_y
        
        # Create skew transformation matrix
        # The shadow should stretch from the root point outward
        skew_factor = random.uniform(0.3, 0.8)  # How much to skew
        
        # Simple approach: create shadow by skewing the object shape
        shadow_layer = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        
        # For each pixel in the object, create corresponding shadow pixels
        for obj_y, obj_x in zip(object_coords[0], object_coords[1]):
            original_alpha = alpha_channel[obj_y, obj_x]
            if original_alpha == 0:
                continue
                
            # Calculate relative position from shadow root
            rel_x = obj_x - center_x
            rel_y = obj_y - bottom_y
            
            # Apply skew transformation: stretch the shadow away from root
            # Create multiple shadow pixels for each object pixel (stretched effect)
            for stretch in range(1, 4):  # Create 3 levels of stretch
                stretch_factor = stretch * skew_factor
                
                # Calculate stretched shadow position
                skewed_x = rel_x + shadow_offset_x * stretch_factor
                skewed_y = rel_y + shadow_offset_y * stretch_factor
                
                # Convert back to absolute coordinates in expanded canvas
                shadow_x = int(shadow_root_x + skewed_x)
                shadow_y = int(shadow_root_y + skewed_y)
                
                # Check bounds
                if 0 <= shadow_x < new_w and 0 <= shadow_y < new_h:
                    # Shadow gets fainter with distance/stretch
                    fade_factor = 1.0 / (stretch + 1)
                    shadow_alpha = int(original_alpha * shadow_intensity * fade_factor)
                    
                    # Only add shadow if it's stronger than existing
                    if shadow_alpha > shadow_layer[shadow_y, shadow_x, 3]:
                        # Random shadow color
                        shadow_color = random.choice([
                            (30, 30, 30),      # Very dark gray
                            (50, 50, 50),      # Dark gray  
                            (40, 40, 50),      # Slightly bluish dark
                            (50, 50, 40),      # Slightly brownish dark
                        ])
                        
                        shadow_layer[shadow_y, shadow_x, :3] = shadow_color
                        shadow_layer[shadow_y, shadow_x, 3] = shadow_alpha
        
        # Apply blur to shadow for realism
        blur_radius = random.randint(4, 8)
        if blur_radius > 0:
            kernel_size = blur_radius * 2 + 1
            # Blur shadow layer
            shadow_layer[:, :, :3] = cv2.GaussianBlur(shadow_layer[:, :, :3], (kernel_size, kernel_size), 0)
            shadow_layer[:, :, 3] = cv2.GaussianBlur(shadow_layer[:, :, 3], (kernel_size, kernel_size), 0)
        
        # Composite: shadow first, then original object on top
        result = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        
        # Place shadow
        shadow_alpha_norm = shadow_layer[:, :, 3].astype(float) / 255.0
        for i in range(3):
            result[:, :, i] = (shadow_alpha_norm * shadow_layer[:, :, i]).astype(np.uint8)
        result[:, :, 3] = shadow_layer[:, :, 3]
        
        # Place original object on top
        obj_alpha_norm = expanded_canvas[:, :, 3].astype(float) / 255.0
        for i in range(3):
            result[:, :, i] = (
                obj_alpha_norm * expanded_canvas[:, :, i] + 
                (1 - obj_alpha_norm) * result[:, :, i]
            ).astype(np.uint8)
        
        # Combine alpha channels (object takes precedence)
        result[:, :, 3] = np.maximum(expanded_canvas[:, :, 3], result[:, :, 3])

        # If caller only wants the shadow layer and offsets, return them instead
        if return_shadow_layer:
            # orig_offset_x/orig_offset_y are the top-left position inside expanded canvas
            return shadow_layer, orig_offset_x, orig_offset_y

        return result
    
    def create_canvas_shadow(self, canvas, object_image, start_x, start_y, shadow_intensity=0.3):
        """
        Create a shadow effect directly on the canvas without changing the object size.
        
        Args:
            canvas (numpy.ndarray): The background canvas to draw shadow on
            object_image (numpy.ndarray): The object image to create shadow from
            start_x (int): X position where object will be placed
            start_y (int): Y position where object will be placed
            shadow_intensity (float): Shadow darkness (0.0-1.0)
        """
        if len(object_image.shape) != 3:
            return  # Can't create shadow from non-RGB image
        
        obj_h, obj_w = object_image.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]
        
        # Random shadow offset (direction and distance)
        shadow_offset_x = random.randint(-20, 20)
        shadow_offset_y = random.randint(15, 40)  # Mostly downward shadows
        
        # 10% chance for upward shadow (different lighting angle)
        if random.random() < 0.1:
            shadow_offset_y = -abs(shadow_offset_y)
        
        # Calculate shadow position on canvas
        shadow_start_x = start_x + shadow_offset_x
        shadow_start_y = start_y + shadow_offset_y
        
        # Ensure shadow stays within canvas bounds
        shadow_start_x = max(0, min(shadow_start_x, canvas_w - obj_w))
        shadow_start_y = max(0, min(shadow_start_y, canvas_h - obj_h))
        shadow_end_x = min(shadow_start_x + obj_w, canvas_w)
        shadow_end_y = min(shadow_start_y + obj_h, canvas_h)
        
        # Calculate actual shadow dimensions within canvas
        actual_shadow_w = shadow_end_x - shadow_start_x
        actual_shadow_h = shadow_end_y - shadow_start_y
        
        if actual_shadow_w <= 0 or actual_shadow_h <= 0:
            return  # Shadow would be outside canvas
        
        # Create object mask for shadow shape
        if len(object_image.shape) == 3 and object_image.shape[2] == 4:
            # RGBA image - use alpha channel
            object_mask = object_image[:actual_shadow_h, :actual_shadow_w, 3] > 0
        else:
            # RGB image - create mask from non-black pixels
            gray_object = cv2.cvtColor(object_image[:actual_shadow_h, :actual_shadow_w], cv2.COLOR_RGB2GRAY)
            object_mask = gray_object > 30  # Threshold to exclude near-black pixels
        
        if not np.any(object_mask):
            return  # No object pixels to create shadow from
        
        # Create shadow by darkening the background
        shadow_color = random.choice([
            (30, 30, 30),      # Very dark gray
            (50, 50, 50),      # Dark gray
            (40, 40, 50),      # Slightly bluish dark
            (50, 50, 40),      # Slightly brownish dark
        ])
        
        # Get the canvas region where shadow will be applied
        canvas_region = canvas[shadow_start_y:shadow_end_y, shadow_start_x:shadow_end_x]
        
        # Apply shadow effect - blend shadow color with background
        for i in range(3):  # RGB channels
            canvas_region[:, :, i] = np.where(
                object_mask,
                (canvas_region[:, :, i] * (1 - shadow_intensity) + 
                 shadow_color[i] * shadow_intensity).astype(np.uint8),
                canvas_region[:, :, i]
            )
        
        # Optional: Add slight blur to shadow for realism
        if random.random() < 0.7:  # 70% chance to blur shadow
            blur_radius = random.randint(2, 5)
            kernel_size = blur_radius * 2 + 1
            
            # Create a temporary shadow layer to blur
            shadow_layer = canvas[shadow_start_y:shadow_end_y, shadow_start_x:shadow_end_x].copy()
            shadow_layer = cv2.GaussianBlur(shadow_layer, (kernel_size, kernel_size), 0)
            
            # Apply blurred shadow only where object mask is true
            for i in range(3):
                canvas[shadow_start_y:shadow_end_y, shadow_start_x:shadow_end_x, i] = np.where(
                    object_mask,
                    shadow_layer[:, :, i],
                    canvas[shadow_start_y:shadow_end_y, shadow_start_x:shadow_end_x, i]
                )
    
    def smooth_edges(self, image, smooth_radius=2):
        """
        Smooth the edges of a PNG object to make it blend more naturally with backgrounds.
        
        This reduces the harsh cutoff effect of PNG objects by applying edge smoothing
        to the alpha channel, creating a more gradual transition at the edges.
        
        Args:
            image (numpy.ndarray): Input RGBA image with transparent background
            smooth_radius (int): Radius for edge smoothing (1-5 pixels)
            
        Returns:
            numpy.ndarray: Image with smoothed edges
        """
        if len(image.shape) != 3 or image.shape[2] != 4:
            # No alpha channel, return original image
            return image
        
        result = image.copy()
        
        # Extract alpha channel
        alpha = result[:, :, 3].astype(float)
        
        # Create edge detection mask to find object boundaries
        # Use morphological operations to find the edge region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_radius*2+1, smooth_radius*2+1))
        
        # Dilated version (expanded object)
        alpha_dilated = cv2.dilate((alpha > 0).astype(np.uint8), kernel, iterations=1) * 255
        
        # Eroded version (shrunk object)  
        alpha_eroded = cv2.erode((alpha > 0).astype(np.uint8), kernel, iterations=1) * 255
        
        # Edge region is the difference between dilated and eroded
        edge_mask = (alpha_dilated.astype(float) - alpha_eroded.astype(float)) / 255.0
        
        # Apply Gaussian blur to the original alpha in the edge region
        alpha_blurred = cv2.GaussianBlur(alpha, (smooth_radius*2+1, smooth_radius*2+1), smooth_radius/3)
        
        # Blend original alpha with blurred alpha based on edge mask
        # In edge regions, use more of the blurred version
        alpha_smoothed = alpha * (1 - edge_mask) + alpha_blurred * edge_mask
        
        # Apply additional subtle smoothing to the entire alpha channel
        if smooth_radius > 1:
            alpha_smoothed = cv2.GaussianBlur(alpha_smoothed, (3, 3), 0.5)
        
        # Ensure alpha values stay in valid range
        alpha_smoothed = np.clip(alpha_smoothed, 0, 255)
        
        # Update result with smoothed alpha
        result[:, :, 3] = alpha_smoothed.astype(np.uint8)
        
        return result
    
    def add_metal_reflection(self, image, reflection_intensity=0.3):
        """
        Add realistic metal reflection effects to make objects look more metallic.
        
        Creates random bright spots and gradients that simulate light reflections
        on metal surfaces, making the objects appear more three-dimensional and realistic.
        
        Args:
            image (numpy.ndarray): Input RGBA image
            reflection_intensity (float): Intensity of reflection (0.0-1.0)
            
        Returns:
            numpy.ndarray: Image with added metal reflection effects
        """
        if len(image.shape) != 3 or image.shape[2] != 4:
            # No alpha channel, return original image
            return image
        
        result = image.copy()
        h, w = image.shape[:2]
        
        # Get the alpha mask to only apply reflections where the object exists
        alpha_mask = (image[:, :, 3] > 0)
        
        if not np.any(alpha_mask):
            return result  # No object pixels, return as-is
        
        # Create reflection overlay
        reflection_overlay = np.zeros((h, w, 3), dtype=np.float32)
        
        # Type of reflection effects to randomly choose from
        reflection_type = random.choice(['spot', 'gradient', 'streak', 'multiple_spots', 'curved_reflection', 'radial_gradient', 'organic_highlight'])
        
        if reflection_type == 'spot':
            # Single bright reflection spot with random shape
            # Find object center
            object_coords = np.where(alpha_mask)
            if len(object_coords[0]) > 0:
                center_y = int(np.mean(object_coords[0]))
                center_x = int(np.mean(object_coords[1]))
                
                # Add some randomness to reflection position
                offset_x = random.randint(-w//4, w//4)
                offset_y = random.randint(-h//4, h//4)
                ref_x = np.clip(center_x + offset_x, 0, w-1)
                ref_y = np.clip(center_y + offset_y, 0, h-1)
                
                # Create elliptical reflection spot with random orientation
                spot_radius_x = random.randint(max(2, min(w, h)//12), max(4, min(w, h)//6))
                spot_radius_y = random.randint(max(2, min(w, h)//12), max(4, min(w, h)//6))
                angle = random.uniform(0, 180)
                
                y_coords, x_coords = np.ogrid[:h, :w]
                
                # Rotate coordinates for ellipse
                angle_rad = np.radians(angle)
                cos_angle = np.cos(angle_rad)
                sin_angle = np.sin(angle_rad)
                
                x_rot = (x_coords - ref_x) * cos_angle + (y_coords - ref_y) * sin_angle
                y_rot = -(x_coords - ref_x) * sin_angle + (y_coords - ref_y) * cos_angle
                
                # Elliptical falloff
                ellipse_mask = np.exp(-((x_rot/spot_radius_x)**2 + (y_rot/spot_radius_y)**2))
                ellipse_mask = np.clip(ellipse_mask * reflection_intensity * random.uniform(200, 300), 0, 255)
                
                # Apply to all color channels
                for i in range(3):
                    reflection_overlay[:, :, i] = ellipse_mask
        
        elif reflection_type == 'gradient':
            # More dynamic gradient with curves and noise
            gradient_direction = random.choice(['horizontal', 'vertical', 'diagonal', 'radial'])
            
            if gradient_direction == 'horizontal':
                gradient = np.linspace(0, 1, w)
                # Add sinusoidal variation
                wave_freq = random.uniform(0.5, 3.0)
                wave_amplitude = random.uniform(0.1, 0.3)
                for i in range(h):
                    wave_offset = np.sin(i * wave_freq / h * 2 * np.pi) * wave_amplitude
                    gradient = np.tile(gradient + wave_offset, (1, 1))
                gradient = np.tile(gradient, (h, 1))
            elif gradient_direction == 'vertical':  
                gradient = np.linspace(0, 1, h)
                # Add sinusoidal variation
                wave_freq = random.uniform(0.5, 3.0)
                wave_amplitude = random.uniform(0.1, 0.3)
                for j in range(w):
                    wave_offset = np.sin(j * wave_freq / w * 2 * np.pi) * wave_amplitude
                    gradient = gradient + wave_offset
                gradient = np.tile(gradient.reshape(-1, 1), (1, w))
            elif gradient_direction == 'radial':
                center_x = random.randint(w//4, 3*w//4)
                center_y = random.randint(h//4, 3*h//4)
                y_coords, x_coords = np.ogrid[:h, :w]
                distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                max_dist = np.sqrt(w**2 + h**2)
                gradient = 1 - (distances / max_dist)
            else:  # diagonal with curve
                x_grad = np.linspace(0, 1, w)
                y_grad = np.linspace(0, 1, h)
                X, Y = np.meshgrid(x_grad, y_grad)
                # Add curved component
                curve_factor = random.uniform(0.5, 2.0)
                gradient = (X + Y + np.sin(X * curve_factor * np.pi) * 0.3) / 2.3
            
            # Add some noise for more organic look
            noise = np.random.normal(0, 0.05, gradient.shape)
            gradient = np.clip(gradient + noise, 0, 1)
            
            # Make it more subtle and varied
            gradient = gradient * reflection_intensity * random.uniform(80, 150)
            gradient = np.clip(gradient, 0, 255)
            
            for i in range(3):
                reflection_overlay[:, :, i] = gradient
        
        elif reflection_type == 'streak':
            # More dynamic light streak with curves
            streak_angle = random.uniform(0, 180)  # Random angle
            streak_width = random.randint(3, max(4, min(w, h)//8))
            
            # Create curved streak path
            center_x, center_y = w//2 + random.randint(-w//6, w//6), h//2 + random.randint(-h//6, h//6)
            y_coords, x_coords = np.ogrid[:h, :w]
            
            # Create curved path using sine wave
            curve_intensity = random.uniform(0.1, 0.5)
            angle_rad = np.radians(streak_angle)
            
            # Base rotated coordinates
            rotated_x = (x_coords - center_x) * np.cos(angle_rad) + (y_coords - center_y) * np.sin(angle_rad)
            rotated_y = -(x_coords - center_x) * np.sin(angle_rad) + (y_coords - center_y) * np.cos(angle_rad)
            
            # Add curve to the streak
            curve_offset = np.sin(rotated_x * curve_intensity) * streak_width * 0.5
            curved_distance = np.abs(rotated_y + curve_offset)
            
            # Create streak with falloff
            streak_mask = np.exp(-(curved_distance**2) / (2 * streak_width**2))
            
            # Add intensity variation along the streak
            intensity_variation = 0.5 + 0.5 * np.sin(rotated_x * 0.1)
            streak_mask *= intensity_variation
            
            streak_mask = np.clip(streak_mask * reflection_intensity * random.uniform(150, 250), 0, 255)
            
            for i in range(3):
                reflection_overlay[:, :, i] = streak_mask
        
        elif reflection_type == 'curved_reflection':
            # Curved reflection like light hitting a cylindrical surface
            object_coords = np.where(alpha_mask)
            if len(object_coords[0]) > 0:
                center_y = int(np.mean(object_coords[0]))
                center_x = int(np.mean(object_coords[1]))
                
                y_coords, x_coords = np.ogrid[:h, :w]
                
                # Create curved reflection pattern
                curve_direction = random.choice(['horizontal', 'vertical'])
                curve_intensity = random.uniform(0.02, 0.08)
                
                if curve_direction == 'horizontal':
                    # Horizontal curve (like cylindrical surface)
                    distance_from_center = np.abs(y_coords - center_y)
                    curve_factor = np.cos(distance_from_center * curve_intensity) * 0.5 + 0.5
                    lateral_falloff = np.exp(-((x_coords - center_x)**2) / (2 * (w//4)**2))
                else:
                    # Vertical curve
                    distance_from_center = np.abs(x_coords - center_x)
                    curve_factor = np.cos(distance_from_center * curve_intensity) * 0.5 + 0.5
                    lateral_falloff = np.exp(-((y_coords - center_y)**2) / (2 * (h//4)**2))
                
                curved_reflection = curve_factor * lateral_falloff
                curved_reflection = np.clip(curved_reflection * reflection_intensity * random.uniform(120, 200), 0, 255)
                
                for i in range(3):
                    reflection_overlay[:, :, i] = curved_reflection
        
        elif reflection_type == 'radial_gradient':
            # Radial gradient from random center point
            object_coords = np.where(alpha_mask)
            if len(object_coords[0]) > 0:
                # Random center point within object bounds
                center_y = int(np.mean(object_coords[0])) + random.randint(-h//6, h//6)
                center_x = int(np.mean(object_coords[1])) + random.randint(-w//6, w//6)
                center_y = np.clip(center_y, 0, h-1)
                center_x = np.clip(center_x, 0, w-1)
                
                y_coords, x_coords = np.ogrid[:h, :w]
                distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                
                # Create radial gradient with random falloff
                max_radius = random.uniform(min(w, h)//8, min(w, h)//3)
                radial_mask = np.exp(-(distances**2) / (2 * max_radius**2))
                
                # Add angular variation for more dynamic look
                angles = np.arctan2(y_coords - center_y, x_coords - center_x)
                angular_variation = 0.7 + 0.3 * np.sin(angles * random.randint(2, 6))
                radial_mask *= angular_variation
                
                radial_mask = np.clip(radial_mask * reflection_intensity * random.uniform(180, 280), 0, 255)
                
                for i in range(3):
                    reflection_overlay[:, :, i] = radial_mask
        
        elif reflection_type == 'organic_highlight':
            # Organic, irregular highlight using Perlin-like noise
            object_coords = np.where(alpha_mask)
            if len(object_coords[0]) > 0:
                y_coords, x_coords = np.ogrid[:h, :w]
                
                # Create multiple layers of sine waves for organic pattern
                organic_pattern = np.zeros((h, w))
                
                for _ in range(random.randint(2, 4)):
                    freq_x = random.uniform(0.05, 0.2)
                    freq_y = random.uniform(0.05, 0.2)
                    phase_x = random.uniform(0, 2*np.pi)
                    phase_y = random.uniform(0, 2*np.pi)
                    amplitude = random.uniform(0.3, 0.7)
                    
                    layer = amplitude * np.sin(x_coords * freq_x + phase_x) * np.sin(y_coords * freq_y + phase_y)
                    organic_pattern += layer
                
                # Normalize and apply
                organic_pattern = (organic_pattern + np.abs(organic_pattern.min())) / (organic_pattern.max() - organic_pattern.min() + 1e-8)
                organic_pattern = np.clip(organic_pattern * reflection_intensity * random.uniform(100, 200), 0, 255)
                
                for i in range(3):
                    reflection_overlay[:, :, i] = organic_pattern
        
        elif reflection_type == 'multiple_spots':
            # Multiple small reflection spots
            num_spots = random.randint(2, 5)
            object_coords = np.where(alpha_mask)
            
            if len(object_coords[0]) > 0:
                for _ in range(num_spots):
                    # Random position within object bounds
                    idx = random.randint(0, len(object_coords[0]) - 1)
                    spot_y = object_coords[0][idx] + random.randint(-h//8, h//8)
                    spot_x = object_coords[1][idx] + random.randint(-w//8, w//8)
                    spot_y = np.clip(spot_y, 0, h-1)
                    spot_x = np.clip(spot_x, 0, w-1)
                    
                    # Small reflection spot
                    spot_radius = random.randint(2, max(3, min(w, h)//12))
                    y_coords, x_coords = np.ogrid[:h, :w]
                    distances = np.sqrt((x_coords - spot_x)**2 + (y_coords - spot_y)**2)
                    
                    spot_mask = np.exp(-(distances**2) / (2 * (spot_radius/2)**2))
                    spot_intensity = reflection_intensity * random.uniform(0.5, 1.0) * 200
                    spot_mask = np.clip(spot_mask * spot_intensity, 0, 255)
                    
                    # Add to existing reflection overlay
                    for i in range(3):
                        reflection_overlay[:, :, i] = np.maximum(reflection_overlay[:, :, i], spot_mask)
        
        # Apply reflection only where object exists
        reflection_overlay = reflection_overlay * alpha_mask[:, :, np.newaxis]
        
        # Blend reflection with original image
        # Use additive blending for realistic metal reflection
        for i in range(3):
            original_channel = result[:, :, i].astype(np.float32)
            reflected_channel = original_channel + reflection_overlay[:, :, i]
            result[:, :, i] = np.clip(reflected_channel, 0, 255).astype(np.uint8)
        
        return result
    
    def generate_noisy_background(self, canvas):
        """
        Generate a noisy background with random shapes, lines, and patterns.
        
        Adds various types of visual noise to the background canvas to improve
        model robustness by teaching it to ignore irrelevant visual elements.
        
        Args:
            canvas (numpy.ndarray): Background canvas to add noise to
            
        Returns:
            numpy.ndarray: Canvas with added noise elements
            
        Note:
            Noise types include:
            - Random circles, rectangles, ellipses, and polygons
            - Random lines and grid patterns  
            - Text elements and pixel noise
            - Texture patterns (grid, dots, diagonal lines)
            
            The amount of noise is controlled by self.noise_intensity (0.0-1.0).
        """
        """Generate a noisy background with random shapes, lines, and patterns"""
        if not self.use_noisy_background:
            return canvas
        
        # Determine noise intensity for this image
        current_noise_intensity = random.uniform(0.1, 0.9) if self.random_noise_intensity else self.noise_intensity
        
        if current_noise_intensity <= 0:
            return canvas
        
        # Create a copy to work with
        noisy_canvas = canvas.copy()
        
        # Calculate number of noise elements based on intensity
        max_noise_elements = int(100 * current_noise_intensity)
        num_noise_elements = random.randint(max_noise_elements // 3, max_noise_elements)
        
        for _ in range(num_noise_elements):
            noise_type = random.choice(['circle', 'rectangle', 'line', 'ellipse', 'polygon', 'text_noise'])
            
            # Random color (but not pure white to maintain contrast)
            color = (
                random.randint(50, 200),
                random.randint(50, 200), 
                random.randint(50, 200)
            )
            
            # Random transparency
            alpha = random.uniform(0.1, 0.6)  # Semi-transparent
            
            if noise_type == 'circle':
                center = (random.randint(0, self.canvas_width), random.randint(0, self.canvas_height))
                radius = random.randint(5, 50)
                thickness = random.choice([-1, 1, 2, 3])  # -1 for filled
                cv2.circle(noisy_canvas, center, radius, color, thickness)
                
            elif noise_type == 'rectangle':
                x1 = random.randint(0, self.canvas_width - 50)
                y1 = random.randint(0, self.canvas_height - 50)
                x2 = x1 + random.randint(10, 100)
                y2 = y1 + random.randint(10, 100)
                thickness = random.choice([-1, 1, 2, 3])
                cv2.rectangle(noisy_canvas, (x1, y1), (x2, y2), color, thickness)
                
            elif noise_type == 'line':
                pt1 = (random.randint(0, self.canvas_width), random.randint(0, self.canvas_height))
                pt2 = (random.randint(0, self.canvas_width), random.randint(0, self.canvas_height))
                thickness = random.randint(1, 5)
                cv2.line(noisy_canvas, pt1, pt2, color, thickness)
                
            elif noise_type == 'ellipse':
                center = (random.randint(0, self.canvas_width), random.randint(0, self.canvas_height))
                axes = (random.randint(10, 80), random.randint(10, 80))
                angle = random.randint(0, 360)
                thickness = random.choice([-1, 1, 2, 3])
                cv2.ellipse(noisy_canvas, center, axes, angle, 0, 360, color, thickness)
                
            elif noise_type == 'polygon':
                # Random polygon with 3-8 points - make smaller polygons
                num_points = random.randint(3, 6)  # Reduced max points for smaller shapes
                
                # Create smaller polygon by constraining size
                center_x = random.randint(50, self.canvas_width - 50)
                center_y = random.randint(50, self.canvas_height - 50)
                max_radius = random.randint(20, self.canvas_height/2)  # Smaller max radius
                
                points = []
                for _ in range(num_points):
                    # Generate points around a center with limited radius
                    angle = random.uniform(0, 2 * np.pi)
                    radius = random.randint(10, max_radius)
                    x = int(center_x + radius * np.cos(angle))
                    y = int(center_y + radius * np.sin(angle))
                    # Ensure points stay within canvas bounds
                    x = max(0, min(x, self.canvas_width))
                    y = max(0, min(y, self.canvas_height))
                    points.append([x, y])
                
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                if random.choice([True, False]):
                    # Filled polygon
                    cv2.fillPoly(noisy_canvas, [pts], color)
                else:
                    # Outline only
                    cv2.polylines(noisy_canvas, [pts], True, color, random.randint(1, 3))
                    
            elif noise_type == 'text_noise':
                # Add random text as noise
                text_samples = ['ABC', '123', 'XYZ', '***', '???', '###', '@@@', '000']
                text = random.choice(text_samples)
                
                position = (random.randint(0, self.canvas_width - 50), random.randint(20, self.canvas_height))
                font_scale = random.uniform(0.5, 2.0)
                thickness = random.randint(1, 3)
                cv2.putText(noisy_canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, color, thickness)
        
        # Add some random noise pixels
        if current_noise_intensity > 0.5:
            # Generate random noise overlay
            noise = np.random.randint(0, 100, (self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            noise_alpha = current_noise_intensity * 0.1  # Very subtle noise
            noisy_canvas = cv2.addWeighted(noisy_canvas, 1 - noise_alpha, noise, noise_alpha, 0)
        
        # Add some texture patterns
        if random.random() < current_noise_intensity:
            self._add_texture_pattern(noisy_canvas, current_noise_intensity)
        
        return noisy_canvas, current_noise_intensity
    
    def _add_texture_pattern(self, canvas, noise_intensity):
        """
        Add subtle texture patterns to the background.
        Args:
            canvas (numpy.ndarray): Canvas to add texture patterns to
            noise_intensity (float): Current noise intensity level
        """
        pattern_type = random.choice(['grid', 'dots', 'lines'])
        
        # Use a subtle color
        pattern_color = (
            random.randint(200, 240),
            random.randint(200, 240),
            random.randint(200, 240)
        )
        
        if pattern_type == 'grid':
            # Add a subtle grid pattern
            grid_spacing = random.randint(50, 150)
            for x in range(0, self.canvas_width, grid_spacing):
                cv2.line(canvas, (x, 0), (x, self.canvas_height), pattern_color, 1)
            for y in range(0, self.canvas_height, grid_spacing):
                cv2.line(canvas, (0, y), (self.canvas_width, y), pattern_color, 1)
                
        elif pattern_type == 'dots':
            # Add random dots
            dot_density = int(50 * noise_intensity)
            for _ in range(dot_density):
                center = (random.randint(0, self.canvas_width), random.randint(0, self.canvas_height))
                radius = random.randint(1, 3)
                cv2.circle(canvas, center, radius, pattern_color, -1)
                
        elif pattern_type == 'lines':
            # Add random diagonal lines
            line_count = int(20 * noise_intensity)
            for _ in range(line_count):
                if random.choice([True, False]):
                    # Horizontal lines
                    y = random.randint(0, self.canvas_height)
                    cv2.line(canvas, (0, y), (self.canvas_width, y), pattern_color, 1)
                else:
                    # Vertical lines
                    x = random.randint(0, self.canvas_width)
                    cv2.line(canvas, (x, 0), (x, self.canvas_height), pattern_color, 1)
    
    def create_background_canvas(self):
        """
        Create background canvas based on selected background mode.
        
        Creates the base canvas for image generation according to the configured
        background mode. Handles RGBA to RGB conversion for transparent backgrounds.
        
        Returns:
            numpy.ndarray: RGB background canvas of size (canvas_height, canvas_width, 3)
            
        Background Mode Behaviors:
            - "blank": Pure white background
            - "random": Random choice between blank and all available backgrounds  
            - "random_backgrounds": Random choice from available backgrounds only
            - Specific name: Uses the named background image
            
        Note:
            RGBA backgrounds are blended with white background using alpha channel.
            All outputs are converted to 3-channel RGB format.
        """
        """Create background canvas based on selected background mode"""
        if self.background_mode == "blank":
            # Create blank white canvas
            canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            canvas.fill(255)  # White background
            print(f"🔲 Using blank white background")
            
        elif self.background_mode in self.background_images:
            # Use specific background image (any loaded background)
            bg_image = self.background_images[self.background_mode].copy()
            # Ensure it's RGB format
            if len(bg_image.shape) == 3 and bg_image.shape[2] == 4:
                # Convert RGBA to RGB by blending with white background
                white_bg = np.ones((bg_image.shape[0], bg_image.shape[1], 3), dtype=np.uint8) * 255
                alpha = bg_image[:, :, 3:4] / 255.0
                canvas = (bg_image[:, :, :3] * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            else:
                canvas = bg_image.copy()
            print(f"🖼️ Using background: {self.background_mode}")
                
        elif self.background_mode == "random":
            # Randomly choose between blank and ALL available backgrounds FOR EACH IMAGE
            blank_option = ["blank"]
            bg_options = list(self.background_images.keys())  # All loaded backgrounds
            all_options = blank_option + bg_options
            selected_option = random.choice(all_options)
            
            if selected_option == "blank":
                canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
                canvas.fill(255)
                print(f"🎲 Randomly selected: blank background")
            else:
                bg_image = self.background_images[selected_option].copy()
                # Handle RGBA to RGB conversion
                if len(bg_image.shape) == 3 and bg_image.shape[2] == 4:
                    white_bg = np.ones((bg_image.shape[0], bg_image.shape[1], 3), dtype=np.uint8) * 255
                    alpha = bg_image[:, :, 3:4] / 255.0
                    canvas = (bg_image[:, :, :3] * alpha + white_bg * (1 - alpha)).astype(np.uint8)
                else:
                    canvas = bg_image.copy()
                print(f"🎲 Randomly selected background: {selected_option}")
                
        elif self.background_mode == "random_backgrounds":
            # Randomly choose from ALL available backgrounds only (no blank) FOR EACH IMAGE
            available_backgrounds = list(self.background_images.keys())  # All loaded backgrounds
            if available_backgrounds:
                selected_bg = random.choice(available_backgrounds)
                bg_image = self.background_images[selected_bg].copy()
                # Handle RGBA to RGB conversion
                if len(bg_image.shape) == 3 and bg_image.shape[2] == 4:
                    white_bg = np.ones((bg_image.shape[0], bg_image.shape[1], 3), dtype=np.uint8) * 255
                    alpha = bg_image[:, :, 3:4] / 255.0
                    canvas = (bg_image[:, :, :3] * alpha + white_bg * (1 - alpha)).astype(np.uint8)
                else:
                    canvas = bg_image.copy()
                print(f"🎲 Randomly selected background: {selected_bg}")
            else:
                # Fallback to blank if no backgrounds available
                canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
                canvas.fill(255)
                print(f"⚠️ No background images available, using blank")
        
        else:
            # Default fallback
            canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            canvas.fill(255)
            print(f"⚠️ Unknown background mode, using blank")
        
        return canvas
    
    def analyze_label_formats(self):
        """
        Analyze and report the format types of all labels in the training data.
        """
        if not hasattr(self, 'training_data'):
            print("❌ No training data loaded yet")
            return
        
        box_count = 0
        polygon_count = 0
        class_distribution = {}
        
        print("\n📊 LABEL FORMAT ANALYSIS")
        print("=" * 50)
        
        for data in self.training_data:
            filename = data['filename']
            labels = data['labels']
            
            # Count classes and formats
            for label in labels:
                class_id = label['class_id']
                label_type = label['type']
                
                class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
                
                if label_type == 'box':
                    box_count += 1
                elif label_type == 'polygon':
                    polygon_count += 1
        
        print(f"Box format labels: {box_count}")
        print(f"Polygon format labels: {polygon_count}")
        print(f"Total labels: {box_count + polygon_count}")
        
        print(f"\nClass distribution:")
        for class_id, count in sorted(class_distribution.items()):
            print(f"   Class {class_id}: {count} objects")
        
        print(f"\nFiles processed: {len(self.training_data)}")
        print("=" * 50)
    
    def rotate_image_and_labels(self, image, labels, angle):
        """
        Rotate image and adjust labels accordingly - handles both box and polygon formats.
        
        Applies rotation transformation to both the image and its corresponding
        YOLO format labels, supporting both bounding box and polygon formats.
        
        Args:
            image (numpy.ndarray): Input image to rotate
            labels (list): List of YOLO format labels
            angle (float): Rotation angle in degrees
            
        Returns:
            tuple[numpy.ndarray, list]: Rotated image and adjusted labels
            
        Note:
            - Coordinates are converted from normalized to pixel space for rotation
            - Rotation matrix accounts for image size changes
            - Labels are converted back to normalized coordinates
            - Supports both 'box' and 'polygon' label types
        """
        """Rotate image and adjust labels accordingly - handles both box and polygon formats"""
        if angle == 0:
            return image, labels
        
        # Get image dimensions
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions after rotation
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Rotate image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(0, 0, 0, 0))
        
        # Rotate labels
        rotated_labels = []
        for label in labels:
            class_id = label['class_id']
            label_type = label['type']
            data = label['data']
            
            if label_type == 'box':
                # Handle box format
                x_center, y_center, width, height = data
                
                # Convert normalized coordinates to pixel coordinates
                x_pixel = x_center * w
                y_pixel = y_center * h
                
                # Apply rotation to center point
                rotated_point = np.dot(rotation_matrix, [x_pixel, y_pixel, 1])
                new_x_pixel = rotated_point[0]
                new_y_pixel = rotated_point[1]
                
                # Convert back to normalized coordinates
                new_x_center = new_x_pixel / new_w
                new_y_center = new_y_pixel / new_h
                new_width = width * (w / new_w)
                new_height = height * (h / new_h)
                
                rotated_labels.append({
                    'type': 'box',
                    'class_id': class_id,
                    'data': [new_x_center, new_y_center, new_width, new_height]
                })
                
            elif label_type == 'polygon':
                # Handle polygon format
                points = data.copy()
                rotated_points = []
                
                # Rotate each point
                for i in range(0, len(points), 2):
                    x_norm = points[i]
                    y_norm = points[i + 1]
                    
                    # Convert to pixel coordinates
                    x_pixel = x_norm * w
                    y_pixel = y_norm * h
                    
                    # Apply rotation
                    rotated_point = np.dot(rotation_matrix, [x_pixel, y_pixel, 1])
                    new_x_pixel = rotated_point[0]
                    new_y_pixel = rotated_point[1]
                    
                    # Convert back to normalized coordinates
                    new_x_norm = new_x_pixel / new_w
                    new_y_norm = new_y_pixel / new_h
                    
                    rotated_points.extend([new_x_norm, new_y_norm])
                
                rotated_labels.append({
                    'type': 'polygon',
                    'class_id': class_id,
                    'data': rotated_points
                })
        
        return rotated_image, rotated_labels
    
    def resize_image_and_labels(self, image, labels, target_size):
        """
        Resize image to fit in target size while maintaining aspect ratio.
        Args:
            image (numpy.ndarray): Input image
            labels (list): List of YOLO format labels
            target_size (tuple): Target size (width, height)
        Returns:
            tuple: Resized image and adjusted labels
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor to fit within target size
        scale_w = target_w / w
        scale_h = target_h / h
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Labels remain the same since they're in normalized coordinates
        return resized_image, labels
    
    def place_object_in_cell(self, canvas, labels_list, obj_data, cell_x, cell_y):
        """
        Place an object in a specific cell of the canvas with original size and realistic effects.
        Args:
            canvas (numpy.ndarray): The background canvas to place the object on
            labels_list (list): List to append adjusted labels
            obj_data (dict): Object data containing 'image' and 'labels'
            cell_x (int): Cell x index
            cell_y (int): Cell y index
        """
        # Random rotation angle
        angle = random.uniform(-30, 30)  # Rotate between -30 and 30 degrees
        
        # Apply random exposure adjustment to each object independently
        random_exposure = random.uniform(-0.4, 0.4)  # Random exposure between -0.4 and +0.4
        adjusted_image = self.adjust_exposure(obj_data['image'], exposure_increase=random_exposure)
        
        # Apply realistic enhancement effects (with probability controls)
        enhanced_image = adjusted_image.copy()
        
        # Add metal reflections (40% chance for metallic appearance)
        if random.random() < 0.4:
            reflection_intensity = random.uniform(0.2, 0.5)
            enhanced_image = self.add_metal_reflection(enhanced_image, reflection_intensity)
            print(f"      ✨ Added metal reflection (intensity: {reflection_intensity:.2f})")
        
        # Smooth edges for better blending (60% chance)
        if random.random() < 0.6:
            smooth_radius = random.randint(1, 3)
            enhanced_image = self.smooth_edges(enhanced_image, smooth_radius)
            print(f"      🔘 Applied edge smoothing (radius: {smooth_radius})")
        
        # Rotate image and labels (keeping original size)
        rotated_image, rotated_labels = self.rotate_image_and_labels(
            enhanced_image, obj_data['labels'], angle
        )
        
        # Use rotated image with original size (no resizing)
        obj_h, obj_w = rotated_image.shape[:2]
        
        # Calculate placement position (center of cell)
        start_x = cell_x * self.cell_size + (self.cell_size - obj_w) // 2
        start_y = cell_y * self.cell_size + (self.cell_size - obj_h) // 2
        
        # Ensure the object fits within canvas bounds
        start_x = max(0, min(start_x, self.canvas_width - obj_w))
        start_y = max(0, min(start_y, self.canvas_height - obj_h))
        
        # Create realistic stretched shadow (70% chance) but composite only the shadow layer to canvas
        if random.random() < 0.7:
            shadow_intensity = random.uniform(0.2, 0.4)
            # Request only the shadow layer and original offsets from create_shadow
            shadow_layer, orig_off_x, orig_off_y = self.create_shadow(rotated_image, shadow_offset_x=None, shadow_offset_y=None, shadow_intensity=shadow_intensity, return_shadow_layer=True)
            if shadow_layer is not None:
                sh_h, sh_w = shadow_layer.shape[:2]
                # The original image was placed at (orig_off_x, orig_off_y) inside the expanded shadow canvas
                # So compute where that expanded canvas's top-left should be placed to align the object at (start_x, start_y)
                expanded_top_left_x = start_x - orig_off_x
                expanded_top_left_y = start_y - orig_off_y

                # Determine intersection with actual canvas
                dst_x1 = max(0, expanded_top_left_x)
                dst_y1 = max(0, expanded_top_left_y)
                dst_x2 = min(self.canvas_width, expanded_top_left_x + sh_w)
                dst_y2 = min(self.canvas_height, expanded_top_left_y + sh_h)

                src_x1 = dst_x1 - expanded_top_left_x
                src_y1 = dst_y1 - expanded_top_left_y
                src_x2 = src_x1 + (dst_x2 - dst_x1)
                src_y2 = src_y1 + (dst_y2 - dst_y1)

                if dst_x2 > dst_x1 and dst_y2 > dst_y1:
                    src_region = shadow_layer[src_y1:src_y2, src_x1:src_x2]
                    dst_region = canvas[dst_y1:dst_y2, dst_x1:dst_x2]

                    # Blend shadow onto canvas using shadow alpha
                    shadow_alpha = src_region[:, :, 3:4].astype(float) / 255.0
                    shadow_rgb = src_region[:, :, :3].astype(float)
                    canvas[dst_y1:dst_y2, dst_x1:dst_x2, :] = (
                        shadow_alpha * shadow_rgb + (1 - shadow_alpha) * dst_region.astype(float)
                    ).astype(np.uint8)
            print(f"      🌑 Added realistic shadow effect (intensity: {shadow_intensity:.2f})")
        
        # Place object on canvas
        if len(rotated_image.shape) == 3 and rotated_image.shape[2] == 4:  # RGBA
            # Handle transparency for RGBA images
            alpha = rotated_image[:, :, 3] / 255.0
            alpha = alpha[:, :, np.newaxis]  # Add dimension for broadcasting
            
            # Blend with background using alpha channel
            foreground = rotated_image[:, :, :3]  # RGB channels
            background = canvas[start_y:start_y+obj_h, start_x:start_x+obj_w, :]
            
            # Alpha blending: result = alpha * foreground + (1-alpha) * background
            canvas[start_y:start_y+obj_h, start_x:start_x+obj_w, :] = \
                (alpha * foreground + (1 - alpha) * background).astype(np.uint8)
                
        else:  # RGB or grayscale
            # Direct placement for RGB images
            if len(rotated_image.shape) == 3:
                canvas[start_y:start_y+obj_h, start_x:start_x+obj_w] = rotated_image
            else:
                # Convert grayscale to RGB if needed
                rotated_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2RGB)
                canvas[start_y:start_y+obj_w, start_x:start_x+obj_h] = rotated_rgb
        
        # Adjust labels for canvas coordinates (only for positive examples with labels)
        # Use simplified coordinate transformation like the original working version
        if obj_data['labels']:  # Only process labels if this is a positive example
            for label in rotated_labels:
                class_id = label['class_id']
                label_type = label['type']
                data = label['data']
                
                if label_type == 'box':
                    # Handle box format
                    x_center, y_center, width, height = data
                    
                    # Convert to canvas coordinates (no scaling since we keep original size)
                    canvas_x_center = (start_x + x_center * obj_w) / self.canvas_width
                    canvas_y_center = (start_y + y_center * obj_h) / self.canvas_height
                    canvas_width = (width * obj_w) / self.canvas_width
                    canvas_height = (height * obj_h) / self.canvas_height
                    
                    # Ensure coordinates are within bounds
                    canvas_x_center = max(0, min(1, canvas_x_center))
                    canvas_y_center = max(0, min(1, canvas_y_center))
                    canvas_width = max(0, min(1, canvas_width))
                    canvas_height = max(0, min(1, canvas_height))
                    
                    labels_list.append({
                        'type': 'box',
                        'class_id': class_id,
                        'data': [canvas_x_center, canvas_y_center, canvas_width, canvas_height]
                    })
                    
                elif label_type == 'polygon':
                    # Handle polygon format
                    points = data.copy()
                    canvas_points = []
                    
                    # Transform each point to canvas coordinates (no scaling)
                    for i in range(0, len(points), 2):
                        x_norm = points[i]
                        y_norm = points[i + 1]
                        
                        # Convert to canvas coordinates
                        canvas_x = (start_x + x_norm * obj_w) / self.canvas_width
                        canvas_y = (start_y + y_norm * obj_h) / self.canvas_height
                        
                        # Ensure coordinates are within bounds
                        canvas_x = max(0, min(1, canvas_x))
                        canvas_y = max(0, min(1, canvas_y))
                        
                        canvas_points.extend([canvas_x, canvas_y])
                    
                    labels_list.append({
                        'type': 'polygon',
                        'class_id': class_id,
                        'data': canvas_points
                    })
    
    def generate_training_image(self, num_objects=None, use_balanced_sampling=True):
        """
        Generate a single training image with objects placed on a background.
        
        Creates a background canvas, applies noise if enabled, and places objects
        using either balanced sampling or random selection. Objects are placed
        in a grid to avoid overlap.
        
        Args:
            num_objects (int, optional): Number of objects to place. If None, 
                random number between 8 and 25 is used.
            use_balanced_sampling (bool): Whether to use balanced class representation
                
        Returns:
            tuple[numpy.ndarray, list]: Generated image and list of YOLO format labels
            
        Note:
            - Objects are placed with random rotations and exposure adjustments
            - Grid collision detection prevents object overlap
            - Background mode determines canvas appearance
        """
        """Generate a single training image with balanced object representation and configurable backgrounds"""
        # Create background canvas based on selected mode
        canvas = self.create_background_canvas()
        
        # Apply noise if enabled (works with any background type)
        if self.use_noisy_background:
            canvas, actual_noise_intensity = self.generate_noisy_background(canvas)
            intensity_text = f"random: {actual_noise_intensity:.2f}" if self.random_noise_intensity else f"{actual_noise_intensity:.2f}"
            if self.background_mode == "blank":
                print(f"   + Added noise to blank background (intensity: {intensity_text})")
            elif self.background_mode in ["H1", "H2", "H3"]:
                print(f"   + Added noise to {self.background_mode} background (intensity: {intensity_text})")
            else:
                print(f"   + Added noise (intensity: {intensity_text})")
        
        # Create list to store all labels
        all_labels = []
        
        # Determine number of objects to place
        if num_objects is None:
            num_objects = random.randint(8, min(25, self.grid_cols * self.grid_rows))
        
        # Select objects with balanced representation
        if use_balanced_sampling and hasattr(self, 'class_distribution'):
            selected_objects = self.select_balanced_objects(num_objects)
            print(f"🎯 Using balanced sampling for {num_objects} objects")
        else:
            # Fallback to random selection
            selected_objects = [random.choice(self.training_data) for _ in range(num_objects)]
            print(f"🎲 Using random sampling for {num_objects} objects")
        
        # Generate random positions for objects
        used_cells = set()
        
        for i, obj_data in enumerate(selected_objects):
            # Try to find an unused cell
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                cell_x = random.randint(0, self.grid_cols - 1)
                cell_y = random.randint(0, self.grid_rows - 1)
                
                if (cell_x, cell_y) not in used_cells:
                    used_cells.add((cell_x, cell_y))
                    
                    # Place object
                    self.place_object_in_cell(canvas, all_labels, obj_data, cell_x, cell_y)
                    break
                
                attempts += 1
            
            if attempts >= 100:
                print(f"⚠️ Could not place object {i+1}, canvas full")
        
        return canvas, all_labels
    
    def draw_bounding_boxes(self, image, labels, class_names=None):
        """
        Draw bounding boxes and polygons on image.
        Args:
            image (numpy.ndarray): Input image to draw on
            labels (list): List of YOLO format labels
            class_names (list, optional): List of class names for labeling
        Returns:
            numpy.ndarray: Image with drawn bounding boxes and polygons
        """
        image_with_annotations = image.copy()
        
        for label in labels:
            class_id = label['class_id']
            label_type = label['type']
            data = label['data']
            
            # Choose color based on type
            if label_type == 'box':
                color = (0, 255, 0)  # Green for boxes
            else:
                color = (255, 0, 255)  # Magenta for polygons
            
            if label_type == 'box':
                # Draw bounding box
                x_center, y_center, width, height = data
                
                # Convert normalized coordinates to pixel coordinates
                x_pixel = int(x_center * self.canvas_width)
                y_pixel = int(y_center * self.canvas_height)
                w_pixel = int(width * self.canvas_width)
                h_pixel = int(height * self.canvas_height)
                
                # Calculate top-left corner
                x1 = int(x_pixel - w_pixel // 2)
                y1 = int(y_pixel - h_pixel // 2)
                x2 = int(x_pixel + w_pixel // 2)
                y2 = int(y_pixel + h_pixel // 2)
                
                # Draw rectangle
                cv2.rectangle(image_with_annotations, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_text = f"Box-{class_id}"
                if class_names and class_id < len(class_names):
                    label_text = f"Box-{class_names[class_id]}"
                
                cv2.putText(image_with_annotations, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                           
            elif label_type == 'polygon':
                # Draw polygon
                points = data
                pixel_points = []
                
                # Convert normalized coordinates to pixel coordinates
                for i in range(0, len(points), 2):
                    x_norm = points[i]
                    y_norm = points[i + 1]
                    x_pixel = int(x_norm * self.canvas_width)
                    y_pixel = int(y_norm * self.canvas_height)
                    pixel_points.append([x_pixel, y_pixel])
                
                # Draw polygon
                pts = np.array(pixel_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image_with_annotations, [pts], True, color, 2)
                
                # Draw label
                label_text = f"Poly-{class_id}"
                if class_names and class_id < len(class_names):
                    label_text = f"Poly-{class_names[class_id]}"
                
                # Place label at first point
                if len(pixel_points) > 0:
                    cv2.putText(image_with_annotations, label_text, 
                               (pixel_points[0][0], pixel_points[0][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return image_with_annotations
    
    def save_training_data(self, image, labels, output_dir="./generated_training", filename_prefix="generated"):
        """
        Save generated training image and labels in their original formats.
        
        Saves the generated image as JPG and labels in YOLO format, preserving
        the original label types (box vs polygon) for maximum compatibility.
        
        Args:
            image (numpy.ndarray): Generated training image
            labels (list): List of YOLO format labels  
            output_dir (str): Directory to save files in
            filename_prefix (str): Prefix for generated filenames
            
        Returns:
            tuple[str, str]: Paths to saved image and label files
            
        Note:
            - Images are saved as JPG files with RGB to BGR conversion
            - Labels preserve their original format (box vs polygon)
            - Filenames include timestamp for uniqueness
            - Output directory is created if it doesn't exist
        """
        """Save generated training image and labels in their original formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_filename = f"{filename_prefix}_{timestamp}.jpg"
        label_filename = f"{filename_prefix}_{timestamp}.txt"
        
        # Save image
        image_path = os.path.join(output_dir, image_filename)
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Save labels in their original formats
        label_path = os.path.join(output_dir, label_filename)
        with open(label_path, 'w') as f:
            for label in labels:
                class_id = label['class_id']
                label_type = label['type']
                data = label['data']
                
                if label_type == 'box':
                    # Save as box format: class x_center y_center width height
                    x_center, y_center, width, height = data
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                elif label_type == 'polygon':
                    # Save as polygon format: class x1 y1 x2 y2 x3 y3 ...
                    line = f"{class_id}"
                    for coord in data:
                        line += f" {coord:.6f}"
                    f.write(line + "\n")
        
        return image_path, label_path
    
    def generate_multiple_images(self, n_images=5, show_results=True, use_balanced_sampling=True):
        """
        Generate multiple training images with balanced class representation.
        
        Creates a specified number of training images, saves them to disk, and
        provides comprehensive statistics about class distribution and balance.
        
        Args:
            n_images (int): Number of images to generate
            show_results (bool): Whether to display first few generated images
            use_balanced_sampling (bool): Use balanced class sampling for better
                representation of underrepresented classes
                
        Returns:
            list[tuple]: List of (image_path, label_path, canvas, labels) tuples
            
        Side Effects:
            - Saves images and labels to ./generated_training/ directory
            - Prints generation statistics and class distribution
            - Shows sample images if show_results=True
            
        Example:
            ```python
            # Generate 50 balanced training images
            files = generator.generate_multiple_images(
                n_images=50,
                use_balanced_sampling=True
            )
            ```
        """
        """Generate multiple training images with balanced class representation and negative examples"""
        print(f"Generating {n_images} training images...")
        if use_balanced_sampling:
            print(f"🎯 Using balanced sampling to ensure equal representation")
            print(f"🔲 Including negative examples naturally (empty label files will contribute backgrounds)")
        
        generated_files = []
        total_class_stats = {}  # Track total class distribution across all generated images
        negative_count = 0  # Track number of images with no objects (negative examples)
        
        for i in range(n_images):
            print(f"Generating image {i+1}/{n_images}...")
            
            # Generate image with balanced sampling
            canvas, labels = self.generate_training_image(use_balanced_sampling=use_balanced_sampling)
            
            # Update total statistics
            if len(labels) == 0:
                negative_count += 1
                print(f"   🔲 Generated negative example (background-only)")
            else:
                for label in labels:
                    class_id = label['class_id']
                    total_class_stats[class_id] = total_class_stats.get(class_id, 0) + 1
            
            # Save files
            image_path, label_path = self.save_training_data(canvas, labels)
            generated_files.append((image_path, label_path, canvas, labels))
            
            # Show results for first few images
            if show_results and i < 3:
                title_suffix = "Background-only" if len(labels) == 0 else "With objects"
                self.show_result(canvas, labels, title=f"Generated Image {i+1} ({title_suffix})")
        
        # Print final balance statistics
        self.print_generation_statistics(total_class_stats, n_images, negative_count)
        
        print(f"\nSuccessfully generated {n_images} training images!")
        print(f"   📁 Positive examples: {n_images - negative_count}")
        print(f"   🔲 Negative examples: {negative_count}")
        print(f"Files saved to: ./generated_training/")
        
        return generated_files
    
    def print_generation_statistics(self, total_class_stats, n_images, negative_count=0):
        """
        Print statistics about generated images including negative examples.
        Args:
            total_class_stats (dict): Dictionary with class_id as key and count as value
            n_images (int): Total number of images generated
            negative_count (int): Number of images with no objects (negative examples)
        """
        print(f"\n📈 GENERATION STATISTICS ({n_images} images)")
        print("=" * 60)
        
        # Class names for display
        class_names = [
            "M4-12mm", "M4-16mm", "M4-20mm", "M4-8mm", "M4-Nut", "M4-Washer",
            "M6-12mm", "M6-16mm", "M6-20mm", "M6-8mm", "M6-Nut", "M6-Washer",
            "Standing-Nut", "Standing-Screw"
        ]
        
        total_objects = sum(total_class_stats.values())
        positive_images = n_images - negative_count
        avg_per_image = total_objects / positive_images if positive_images > 0 else 0
        
        print(f"Total images generated: {n_images}")
        print(f"   📁 Positive images (with objects): {positive_images}")
        print(f"   🔲 Negative images (backgrounds only): {negative_count}")
        print(f"Total objects generated: {total_objects}")
        print(f"Average objects per positive image: {avg_per_image:.1f}")
        
        if total_class_stats:
            print(f"\nClass distribution in generated images:")
        
        # Calculate balance metrics
        counts = list(total_class_stats.values())
        max_count = max(counts) if counts else 0
        min_count = min(counts) if counts else 0
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        for class_id in sorted(total_class_stats.keys()):
            count = total_class_stats[class_id]
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            
            # Balance indicator
            balance_indicator = "✅" if count >= max_count * 0.7 else "⚠️" if count >= max_count * 0.3 else "❌"
            
            print(f"   Class {class_id:2d} ({class_name:15s}): {count:4d} objects ({percentage:5.1f}%) {balance_indicator}")
        
        print(f"\nBalance metrics:")
        print(f"   Most frequent class: {max_count} objects")
        print(f"   Least frequent class: {min_count} objects")
        print(f"   Balance ratio: {balance_ratio:.2f} (1.0 = perfect balance)")
        
        if balance_ratio >= 0.8:
            print(f"   🎉 Excellent balance achieved!")
        elif balance_ratio >= 0.6:
            print(f"   ✅ Good balance achieved!")
        elif balance_ratio >= 0.4:
            print(f"   ⚠️ Moderate balance - consider adjusting")
        else:
            print(f"   ❌ Poor balance - check data distribution")
        
        if negative_count > 0:
            negative_percentage = (negative_count / n_images * 100)
            print(f"\n🔲 Negative examples: {negative_count} images ({negative_percentage:.1f}%)")
            print(f"   This helps the model learn to distinguish backgrounds from objects")
        
        print("=" * 60)
    
    def generate_comparison_images(self, n_images=3):
        """
        Generate comparison images with different noise levels for evaluation.
        Args:
            n_images (int): Number of images to generate for each noise level
        Returns:
            list[tuple]: List of (image_path, label_path, canvas, labels, noise_level) tuples
        """
        print(f"Generating comparison images with different noise levels...")
        
        original_use_noise = self.use_noisy_background
        original_intensity = self.noise_intensity
        
        noise_levels = [
            (False, 0.0, "clean"),
            (True, 0.2, "light_noise"),
            (True, 0.5, "medium_noise"),
            (True, 0.8, "heavy_noise")
        ]
        
        comparison_files = []
        
        for level_idx, (use_noise, intensity, level_name) in enumerate(noise_levels):
            print(f"Generating {level_name} images (noise: {intensity})...")
            
            # Update noise settings
            self.use_noisy_background = use_noise
            self.noise_intensity = intensity
            
            for i in range(n_images):
                # Generate image
                canvas, labels = self.generate_training_image()
                
                # Save with specific naming
                filename_prefix = f"comparison_{level_name}"
                image_path, label_path = self.save_training_data(canvas, labels, 
                                                               filename_prefix=filename_prefix)
                comparison_files.append((image_path, label_path, canvas, labels, level_name))
                
                # Show first image of each noise level
                if i == 0:
                    self.show_result(canvas, labels, 
                                   title=f"Comparison: {level_name.replace('_', ' ').title()} (Intensity: {intensity})")
        
        # Restore original settings
        self.use_noisy_background = original_use_noise
        self.noise_intensity = original_intensity
        
        print(f"\nGenerated {len(comparison_files)} comparison images!")
        return comparison_files
    
    def show_result(self, image, labels, title="Generated Training Image"):
        """
        Display the generated image with and without annotations.
        Args:
            image (numpy.ndarray): Generated training image
            labels (list): List of YOLO format labels
            title (str): Title for the display window
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Show image without annotations
        ax1.imshow(image)
        ax1.set_title(f"{title} - Original", fontsize=14)
        ax1.axis('off')
        
        # Show image with annotations (boxes and polygons)
        image_with_annotations = self.draw_bounding_boxes(image, labels)
        ax2.imshow(image_with_annotations)
        ax2.set_title(f"{title} - With Annotations", fontsize=14)
        ax2.axis('off')
        
        # Add statistics
        box_count = sum(1 for label in labels if label['type'] == 'box')
        polygon_count = sum(1 for label in labels if label['type'] == 'polygon')
        stats_text = f"Objects: {len(labels)} ({box_count} boxes, {polygon_count} polygons)\nCanvas: {self.canvas_width}x{self.canvas_height}\nGrid: {self.grid_cols}x{self.grid_rows}"
        fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.show()
    
    def select_balanced_objects(self, num_objects):
        """
        Select objects with balanced representation of all classes AND 50/50 positive/negative balance.
        
        Implements intelligent sampling strategy that ensures:
        1. 50/50 split between positive (with objects) and negative (background-only) examples
        2. Balanced representation of all classes within positive examples
        3. Prioritization of underrepresented classes
        
        Args:
            num_objects (int): Total number of objects to select
            
        Returns:
            list: Selected training data objects with balanced representation
            
        Note:
            - Exactly 50% negative examples (background-only)
            - Remaining 50% distributed across all classes
            - Underrepresented classes are prioritized for better balance
            - Selection is shuffled to randomize placement order
        """
        """Select objects with balanced representation of all classes AND 50/50 positive/negative balance"""
        selected_objects = []
        # Track selected file indices to avoid duplicates within one generated image
        selected_file_indices = set()
        
        # Calculate exact 50/50 split
        target_negatives = num_objects // 2  # Exactly 50% negatives
        target_positives = num_objects - target_negatives  # Remaining are positives
        
        print(f"🎯 Target balance: {target_positives} positives + {target_negatives} negatives = {num_objects} total")
        
        # First: Add negative examples (exactly target_negatives)
        negative_indices = [i for i, data in enumerate(self.training_data) if data['type'] == 'negative']
        if negative_indices and target_negatives > 0:
            # If enough negatives, sample without replacement
            if len(negative_indices) >= target_negatives:
                chosen = random.sample(negative_indices, target_negatives)
                for idx in chosen:
                    selected_objects.append(self.training_data[idx])
                    selected_file_indices.add(idx)
            else:
                # Fallback: allow replacement but avoid immediate duplicates where possible
                for _ in range(target_negatives):
                    idx = random.choice(negative_indices)
                    selected_objects.append(self.training_data[idx])
                    selected_file_indices.add(idx)
            print(f"🔲 Added {target_negatives} negative examples")
        
        # Second: Add positive examples with class balance (exactly target_positives)
        if target_positives > 0:
            # Create a usage counter for each class
            class_usage = {class_id: 0 for class_id in self.class_distribution.keys()}
            
            # Calculate target representation for each class among positive examples
            total_classes = len(self.class_distribution)
            objects_per_class = max(1, target_positives // total_classes)
            remaining_objects = target_positives - (objects_per_class * total_classes)
            
            print(f"   Positive distribution: {objects_per_class} objects per class, {remaining_objects} extra")
            
            # Ensure minimum representation for all classes
            for class_id in self.class_distribution.keys():
                # Work on a copy of candidate file indices to avoid reusing the same file
                candidate_files = list(self.class_to_files[class_id])
                for _ in range(min(objects_per_class, target_positives - len([obj for obj in selected_objects if obj['type'] == 'positive']))):
                    chosen_idx = None
                    # Try to pick an index not already selected
                    available = [idx for idx in candidate_files if idx not in selected_file_indices]
                    if available:
                        chosen_idx = random.choice(available)
                    elif candidate_files:
                        # No available new files, pick any (allow replacement)
                        chosen_idx = random.choice(candidate_files)
                    if chosen_idx is not None:
                        selected_objects.append(self.training_data[chosen_idx])
                        selected_file_indices.add(chosen_idx)
                        class_usage[class_id] += 1
            
            # Distribute remaining positive objects
            remaining_positive_slots = target_positives - len([obj for obj in selected_objects if obj['type'] == 'positive'])
            for _ in range(remaining_positive_slots):
                # Prioritize underrepresented classes
                weighted_classes = []
                for class_id in self.class_distribution.keys():
                    weight = 3.0 if class_id in self.underrepresented_classes else 1.0
                    
                    # Further adjust weight based on current usage
                    current_usage = class_usage[class_id]
                    if current_usage == 0:
                        weight *= 2.0
                    
                    for _ in range(int(weight * 10)):
                        weighted_classes.append(class_id)
                
                if weighted_classes:
                    selected_class = random.choice(weighted_classes)
                    candidate_files = list(self.class_to_files[selected_class])
                    chosen_idx = None
                    available = [idx for idx in candidate_files if idx not in selected_file_indices]
                    if available:
                        chosen_idx = random.choice(available)
                    elif candidate_files:
                        chosen_idx = random.choice(candidate_files)
                    if chosen_idx is not None:
                        selected_objects.append(self.training_data[chosen_idx])
                        selected_file_indices.add(chosen_idx)
                        class_usage[selected_class] += 1
        
        # Shuffle to randomize placement order
        random.shuffle(selected_objects)
        
        # Report selection statistics
        selection_stats = {}
        negative_count = 0
        positive_count = 0
        for obj in selected_objects:
            if obj['type'] == 'negative':
                negative_count += 1
            else:
                positive_count += 1
                for label in obj['labels']:
                    class_id = label['class_id']
                    selection_stats[class_id] = selection_stats.get(class_id, 0) + 1
        
        print(f"📊 FINAL BALANCE:")
        print(f"   Positive examples: {positive_count} ({positive_count/len(selected_objects)*100:.1f}%)")
        print(f"   Negative examples: {negative_count} ({negative_count/len(selected_objects)*100:.1f}%)")
        print(f"   Selected objects by class: {dict(sorted(selection_stats.items()))}")
        
        return selected_objects

    # ...existing code...
def main():
    """
    Main function to run the training image generator with example configuration.
    
    Demonstrates usage of the TrainingImageGenerator with various configuration options
    including background modes, noise settings, and balanced sampling. Generates
    training images and optionally creates comparison images with different noise levels.
    
    Configuration Options:
        - Background modes: "blank", "random", "random_backgrounds", or specific names
        - Noise settings: intensity levels from 0.0 (no noise) to 1.0 (heavy noise)  
        - Balanced sampling: ensures equal class representation
        - Comparison mode: generates images with different noise levels for evaluation
        
    Example Output:
        - 87 training images by default
        - Comprehensive statistics about class distribution
        - Files saved to ./generated_training/ directory
        
    Raises:
        Exception: If training data cannot be loaded or generation fails
    """
    """Main function to run the training image generator"""
    try:
        # Configuration
        use_noisy_background = True  # Set to True to enable noisy backgrounds
        noise_intensity = 1.0  # 0.0 to 1.0, controls how noisy the background is
        
        # Background configuration
        # Base canvas options: "blank", specific background names (e.g., "H1", "H2", "H3"), "random", "random_backgrounds"
        # - "blank": Pure white background
        # - Any background name: Use specific background image (e.g., "H1", "H2", "H3", or any other PNG file in backgrounds directory)
        # - "random": Randomly choose between blank and ALL available background images
        # - "random_backgrounds": Randomly choose only from ALL available background images (no blank)
        # Noise is applied separately based on use_noisy_background setting
        background_mode = "random"  # Change this to select background type

        # Initialize generator
        generator = TrainingImageGenerator(
            images_dir=r"C:\My Files\Uni\Projekt Robotik und Bildverarbeitung\05_Kit_Verificator\Screws_Png\20250715-PngScrews.v1i.yolov8\train\images",
            labels_dir=r"C:\My Files\Uni\Projekt Robotik und Bildverarbeitung\05_Kit_Verificator\Screws_Png\20250715-PngScrews.v1i.yolov8\train\labels",
            use_noisy_background=use_noisy_background,
            noise_intensity=noise_intensity,
            background_mode=background_mode,
            backgrounds_dir="./echteHintergründe",
            random_noise_intensity=True
        )
        
        print("Training Image Generator initialized successfully!")
        print(f"Available training objects: {len(generator.training_data)}")
        print(f"Background mode: {background_mode}")
        print(f"Available backgrounds: {list(generator.background_images.keys())}")
        print(f"Noisy background: {'Enabled' if use_noisy_background else 'Disabled'}")
        if use_noisy_background:
            print(f"Noise intensity: {noise_intensity:.1f}")
        
        # Analyze label formats
        generator.analyze_label_formats()
        
        # Generate training images
        n_images = 3  # Change this to generate more images
        
        # Option to generate comparison images with different noise levels
        generate_comparison = False  # Set to True to generate comparison images
        
        if generate_comparison:
            print("\n" + "="*60)
            print("GENERATING COMPARISON IMAGES")
            print("="*60)
            comparison_files = generator.generate_comparison_images(n_images=2)
            print(f"Generated {len(comparison_files)} comparison images")
        
        generated_files = generator.generate_multiple_images(n_images=n_images, 
                                                           show_results=True, 
                                                           use_balanced_sampling=True)  # Enable balanced sampling
        
        # Print summary
        print("\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        print(f"Generated {len(generated_files)} training images")
        print(f"Output directory: ./generated_training/")
        print(f"Canvas size: {generator.canvas_width}x{generator.canvas_height}")
        print(f"Grid size: {generator.grid_cols}x{generator.grid_rows}")
        print(f"Cell size: {generator.cell_size}x{generator.cell_size}")
        print(f"Background mode: {generator.background_mode}")
        print(f"Available backgrounds: {list(generator.background_images.keys())}")
        print(f"Noisy background: {'Enabled' if generator.use_noisy_background else 'Disabled'}")
        if generator.use_noisy_background:
            print(f"Noise intensity: {generator.noise_intensity:.1f}")
            print("   Random shapes, lines, patterns added to help model ignore irrelevant objects")
        
        # Show file list
        print(f"\nGenerated files:")
        for i, (img_path, label_path, _, labels) in enumerate(generated_files, 1):
            print(f"  {i}. {os.path.basename(img_path)} ({len(labels)} objects)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()