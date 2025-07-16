import os
import cv2
import numpy as np
import random
import math
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

class TrainingImageGenerator:
    def __init__(self, images_dir="./train/images", labels_dir="./train/labels", use_noisy_background=False, noise_intensity=0.3):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.canvas_width = 3076
        self.canvas_height = 1852
        self.cell_size = 250
        
        # Noise background settings
        self.use_noisy_background = use_noisy_background
        self.noise_intensity = noise_intensity  # 0.0 to 1.0, controls how noisy the background is
        
        # Calculate grid dimensions
        self.grid_cols = self.canvas_width // self.cell_size
        self.grid_rows = self.canvas_height // self.cell_size
        
        print(f"Canvas size: {self.canvas_width}x{self.canvas_height}")
        print(f"Grid size: {self.grid_cols}x{self.grid_rows} cells")
        print(f"Cell size: {self.cell_size}x{self.cell_size}")
        
        # Load available PNG files and their labels
        self.load_training_data()
    
    def load_training_data(self):
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
                    if labels:
                        self.training_data.append({
                            'image': image,
                            'labels': labels,
                            'filename': png_file.name
                        })
                        print(f"  Added {len(labels)} labels")
                    else:
                        print(f"  No valid labels found")
                else:
                    print(f"  Could not load image")
            else:
                print(f"Missing label file for {png_file.name}")
        
        print(f"\n📊 Summary: Loaded {len(self.training_data)} training images with labels")
        
        if len(self.training_data) == 0:
            print("No training data found!")
            print("Please check:")
            print(f"   - PNG files are in: {self.images_dir}")
            print(f"   - Label files are in: {self.labels_dir}")
            print(f"   - Label files have same name as PNG files (but .txt extension)")
            raise ValueError("No training data found! Please check your paths.")
    
    def load_yolo_labels(self, label_file):
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
        """Adjust exposure/brightness of an image, preserving alpha channel if present"""
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
    
    def generate_noisy_background(self, canvas):
        """Generate a noisy background with random shapes, lines, and patterns"""
        if not self.use_noisy_background or self.noise_intensity <= 0:
            return canvas
        
        # Create a copy to work with
        noisy_canvas = canvas.copy()
        
        # Calculate number of noise elements based on intensity
        max_noise_elements = int(100 * self.noise_intensity)
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
                # Random polygon with 3-8 points
                num_points = random.randint(3, 8)
                points = []
                for _ in range(num_points):
                    x = random.randint(0, self.canvas_width)
                    y = random.randint(0, self.canvas_height)
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
        if self.noise_intensity > 0.5:
            # Generate random noise overlay
            noise = np.random.randint(0, 100, (self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            noise_alpha = self.noise_intensity * 0.1  # Very subtle noise
            noisy_canvas = cv2.addWeighted(noisy_canvas, 1 - noise_alpha, noise, noise_alpha, 0)
        
        # Add some texture patterns
        if random.random() < self.noise_intensity:
            self._add_texture_pattern(noisy_canvas)
        
        return noisy_canvas
    
    def _add_texture_pattern(self, canvas):
        """Add subtle texture patterns to the background"""
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
            dot_density = int(50 * self.noise_intensity)
            for _ in range(dot_density):
                center = (random.randint(0, self.canvas_width), random.randint(0, self.canvas_height))
                radius = random.randint(1, 3)
                cv2.circle(canvas, center, radius, pattern_color, -1)
                
        elif pattern_type == 'lines':
            # Add random diagonal lines
            line_count = int(20 * self.noise_intensity)
            for _ in range(line_count):
                if random.choice([True, False]):
                    # Horizontal lines
                    y = random.randint(0, self.canvas_height)
                    cv2.line(canvas, (0, y), (self.canvas_width, y), pattern_color, 1)
                else:
                    # Vertical lines
                    x = random.randint(0, self.canvas_width)
                    cv2.line(canvas, (x, 0), (x, self.canvas_height), pattern_color, 1)
    
    def analyze_label_formats(self):
        """Analyze and report the format types of all labels"""
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
        """Resize image to fit in target size while maintaining aspect ratio"""
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
        """Place an object in a specific cell of the canvas with original size"""
        # Random rotation angle
        angle = random.uniform(-30, 30)  # Rotate between -30 and 30 degrees
        
        # Apply random exposure adjustment to each object independently
        random_exposure = random.uniform(-0.4, 0.4)  # Random exposure between -0.2 and +0.2
        adjusted_image = self.adjust_exposure(obj_data['image'], exposure_increase=random_exposure)
        
        # Rotate image and labels (keeping original size)
        rotated_image, rotated_labels = self.rotate_image_and_labels(
            adjusted_image, obj_data['labels'], angle
        )
        
        # Use rotated image with original size (no resizing)
        obj_h, obj_w = rotated_image.shape[:2]
        
        # Calculate placement position (center of cell)
        start_x = cell_x * self.cell_size + (self.cell_size - obj_w) // 2
        start_y = cell_y * self.cell_size + (self.cell_size - obj_h) // 2
        
        # Ensure the object fits within canvas bounds
        start_x = max(0, min(start_x, self.canvas_width - obj_w))
        start_y = max(0, min(start_y, self.canvas_height - obj_h))
        
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
                canvas[start_y:start_y+obj_h, start_x:start_x+obj_w] = rotated_rgb
        
        # Adjust labels for canvas coordinates
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
    
    def generate_training_image(self, num_objects=None):
        """Generate a single training image with random objects and optional noisy background"""
        # Create blank canvas with white background
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        canvas.fill(255)  # Complete white background (was 240 - light gray)
        
        # Add noisy background if enabled
        canvas = self.generate_noisy_background(canvas)
        
        # Create list to store all labels
        all_labels = []
        
        # Determine number of objects to place
        if num_objects is None:
            num_objects = random.randint(5, min(20, self.grid_cols * self.grid_rows))
        
        # Generate random positions for objects
        used_cells = set()
        
        for _ in range(num_objects):
            # Try to find an unused cell
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                cell_x = random.randint(0, self.grid_cols - 1)
                cell_y = random.randint(0, self.grid_rows - 1)
                
                if (cell_x, cell_y) not in used_cells:
                    used_cells.add((cell_x, cell_y))
                    
                    # Select random object
                    obj_data = random.choice(self.training_data)
                    
                    # Place object
                    self.place_object_in_cell(canvas, all_labels, obj_data, cell_x, cell_y)
                    break
                
                attempts += 1
        
        return canvas, all_labels
    
    def draw_bounding_boxes(self, image, labels, class_names=None):
        """Draw bounding boxes and polygons on image"""
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
    
    def generate_multiple_images(self, n_images=5, show_results=True):
        """Generate multiple training images"""
        print(f"Generating {n_images} training images...")
        
        generated_files = []
        
        for i in range(n_images):
            print(f"Generating image {i+1}/{n_images}...")
            
            # Generate image
            canvas, labels = self.generate_training_image()
            
            # Save files
            image_path, label_path = self.save_training_data(canvas, labels)
            generated_files.append((image_path, label_path, canvas, labels))
            
            # Show results for first few images
            if show_results and i < 3:
                self.show_result(canvas, labels, title=f"Generated Image {i+1}")
        
        print(f"\nSuccessfully generated {n_images} training images!")
        print(f"Files saved to: ./generated_training/")
        
        return generated_files
    
    def generate_comparison_images(self, n_images=3):
        """Generate comparison images with different noise levels for evaluation"""
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
        """Display the generated image with and without annotations"""
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


def main():
    """Main function to run the training image generator"""
    try:
        # Configuration
        use_noisy_background = True  # Set to True to enable noisy backgrounds
        noise_intensity = 0.4  # 0.0 to 1.0, controls how noisy the background is
        
        # Initialize generator
        generator = TrainingImageGenerator(
            images_dir="./train/images",
            labels_dir="./train/labels",
            use_noisy_background=use_noisy_background,
            noise_intensity=noise_intensity
        )
        
        print("Training Image Generator initialized successfully!")
        print(f"Available training objects: {len(generator.training_data)}")
        print(f"Noisy background: {'Enabled' if use_noisy_background else 'Disabled'}")
        if use_noisy_background:
            print(f"Noise intensity: {noise_intensity:.1f}")
        
        # Analyze label formats
        generator.analyze_label_formats()
        
        # Generate training images
        n_images = 5  # Change this to generate more images
        
        # Option to generate comparison images with different noise levels
        generate_comparison = False  # Set to True to generate comparison images
        
        if generate_comparison:
            print("\n" + "="*60)
            print("GENERATING COMPARISON IMAGES")
            print("="*60)
            comparison_files = generator.generate_comparison_images(n_images=2)
            print(f"Generated {len(comparison_files)} comparison images")
        
        generated_files = generator.generate_multiple_images(n_images=n_images, show_results=True)
        
        # Print summary
        print("\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        print(f"Generated {len(generated_files)} training images")
        print(f"Output directory: ./generated_training/")
        print(f"Canvas size: {generator.canvas_width}x{generator.canvas_height}")
        print(f"Grid size: {generator.grid_cols}x{generator.grid_rows}")
        print(f"Cell size: {generator.cell_size}x{generator.cell_size}")
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