import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import json

class ModelEvaluationInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Model Evaluation Interface")
        self.root.geometry("1400x900")
        
        # Variables
        self.model = None
        self.model_path = ""
        self.image_path = ""
        self.label_path = ""
        self.image = None
        self.results = None
        self.ground_truth = None
        
        # Dataset variables
        self.dataset_path = ""
        self.dataset_images = []
        self.current_image_index = 0
        
        # Class names for your dataset
        self.class_names = [
            "M4-12mm", "M4-16mm", "M4-20mm", "M4-8mm", "M4-Nut", "M4-Washer",
            "M6-12mm", "M6-16mm", "M6-20mm", "M6-8mm", "M6-Nut", "M6-Washer",
            "Standing-Nut", "Standing-Screw"
        ]
        
        self.setup_ui()
        
    def get_class_colors(self):
        """Get colors for classes, initialize if not already done"""
        if not hasattr(self, 'class_colors'):
            # Fallback color initialization if model not loaded yet
            self.class_colors = plt.cm.Set1(np.linspace(0, 1, len(self.class_names)))
        return self.class_colors

    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # Model selection
        ttk.Label(control_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.model_label = ttk.Label(control_frame, text="No model selected", foreground="red")
        self.model_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(control_frame, text="Select Model", command=self.select_model).grid(row=0, column=2, padx=(5, 0))
        
        # Image selection
        ttk.Label(control_frame, text="Image:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.image_label = ttk.Label(control_frame, text="No image selected", foreground="red")
        self.image_label.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))
        ttk.Button(control_frame, text="Select Image", command=self.select_image).grid(row=1, column=2, padx=(5, 0), pady=(5, 0))
        
        # Label file selection (optional)
        ttk.Label(control_frame, text="Labels (optional):").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.label_label = ttk.Label(control_frame, text="No label file selected", foreground="gray")
        self.label_label.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))
        ttk.Button(control_frame, text="Select Labels", command=self.select_labels).grid(row=2, column=2, padx=(5, 0), pady=(5, 0))
        
        # Run evaluation button
        self.run_button = ttk.Button(control_frame, text="Run Evaluation", command=self.run_evaluation, state="disabled")
        self.run_button.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # Use pre-trained model button (fallback option)
        self.pretrained_button = ttk.Button(control_frame, text="Use Pre-trained YOLOv8n", command=self.use_pretrained_model)
        self.pretrained_button.grid(row=3, column=2, pady=(10, 0), padx=(5, 0))
        
        # Dataset browsing controls
        ttk.Label(control_frame, text="Dataset (optional):").grid(row=4, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.dataset_label = ttk.Label(control_frame, text="No dataset selected", foreground="gray")
        self.dataset_label.grid(row=4, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))
        ttk.Button(control_frame, text="Select Dataset", command=self.select_dataset).grid(row=4, column=2, padx=(5, 0), pady=(5, 0))
        
        # Navigation buttons
        nav_frame = ttk.Frame(control_frame)
        nav_frame.grid(row=5, column=0, columnspan=3, pady=(5, 0), sticky=(tk.W, tk.E))
        
        self.prev_button = ttk.Button(nav_frame, text="◀ Previous", command=self.previous_image, state="disabled")
        self.prev_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.next_button = ttk.Button(nav_frame, text="Next ▶", command=self.next_image, state="disabled")
        self.next_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.image_info_label = ttk.Label(nav_frame, text="")
        self.image_info_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.conf_var = tk.DoubleVar(value=0.3)
        conf_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.conf_var, orient=tk.HORIZONTAL, length=200)
        conf_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        self.conf_label = ttk.Label(settings_frame, text="0.30")
        self.conf_label.grid(row=0, column=2, padx=(5, 0))
        
        # Update confidence label when scale changes
        def update_conf_label(*args):
            self.conf_label.config(text=f"{self.conf_var.get():.2f}")
        self.conf_var.trace('w', update_conf_label)
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Predictions tab
        self.pred_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pred_frame, text="Model Predictions")
        
        # Ground truth tab
        self.gt_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.gt_frame, text="Ground Truth")
        
        # Comparison tab
        self.comp_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comp_frame, text="Comparison")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def use_pretrained_model(self):
        """Use a pre-trained YOLO model as fallback"""
        try:
            self.status_var.set("Loading pre-trained YOLOv8n model...")
            self.root.update()
            
            # Use YOLOv8n which is more stable
            self.model_path = "yolov8n.pt"
            self.model_label.config(text="yolov8n.pt (pre-trained)", foreground="blue")
            self.status_var.set("Pre-trained model selected")
            self.update_run_button_state()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set pre-trained model: {str(e)}")
            
    def select_model(self):
        """Select YOLO model file"""
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[
                ("PyTorch models", "*.pt"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.model_path = file_path
            self.model_label.config(text=Path(file_path).name, foreground="green")
            self.status_var.set(f"Model selected: {Path(file_path).name}")
            self.update_run_button_state()
            
    def select_image(self):
        """Select image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.image_label.config(text=Path(file_path).name, foreground="green")
            self.status_var.set(f"Image selected: {Path(file_path).name}")
            self.update_run_button_state()
            
    def select_labels(self):
        """Select label file (YOLO format)"""
        file_path = filedialog.askopenfilename(
            title="Select Label File",
            filetypes=[
                ("Label files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.label_path = file_path
            self.label_label.config(text=Path(file_path).name, foreground="green")
            self.status_var.set(f"Label file selected: {Path(file_path).name}")
        else:
            self.label_path = ""
            self.label_label.config(text="No label file selected", foreground="gray")
    
    def select_dataset(self):
        """Select dataset YAML file"""
        file_path = filedialog.askopenfilename(
            title="Select Dataset YAML",
            filetypes=[
                ("YAML files", "*.yaml *.yml"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.dataset_path = file_path
            self.load_dataset()
    
    def load_dataset(self):
        """Load dataset from YAML file"""
        try:
            import yaml
            
            with open(self.dataset_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
            
            # Get dataset root directory
            dataset_dir = Path(self.dataset_path).parent
            print(f"Dataset YAML location: {self.dataset_path}")
            print(f"Dataset directory: {dataset_dir}")
            
            # Try multiple paths in order of preference: val, test, train
            possible_paths = []
            for key in ['val', 'test', 'train']:
                if key in dataset_config:
                    possible_paths.append((key, dataset_config[key]))
            
            if not possible_paths:
                messagebox.showerror("Error", "Could not find val, test, or train path in YAML file")
                return
            
            # Try each path until we find one that exists and has images
            images_dir = None
            used_key = None
            
            for key, path in possible_paths:
                print(f"\nTrying {key} with path: {path}")
                
                # Handle different path formats
                possible_dirs = []
                
                if path.startswith('../'):
                    # Relative path - try multiple interpretations
                    # Method 1: Relative to YAML file location
                    dir1 = (dataset_dir / path).resolve()
                    possible_dirs.append(dir1)
                    
                    # Method 2: Maybe the path is meant to be relative to current working directory
                    dir2 = Path(path).resolve()
                    possible_dirs.append(dir2)
                    
                    # Method 3: Remove the ../ and look in dataset directory
                    clean_path = path.replace('../', '')
                    dir3 = dataset_dir / clean_path
                    possible_dirs.append(dir3)
                    
                else:
                    # Absolute path or simple relative path
                    if Path(path).is_absolute():
                        possible_dirs.append(Path(path))
                    else:
                        possible_dirs.append(dataset_dir / path)
                
                # Also try looking for the folder name directly in dataset directory
                folder_name = Path(path).name  # gets 'images' from '../valid/images'
                parent_name = Path(path).parent.name  # gets 'valid' from '../valid/images'
                
                # Try direct subfolder in dataset directory
                if parent_name != '.':
                    possible_dirs.append(dataset_dir / parent_name / folder_name)
                possible_dirs.append(dataset_dir / folder_name)
                
                print(f"  Checking possible directories:")
                for i, test_dir in enumerate(possible_dirs, 1):
                    print(f"    {i}. {test_dir}")
                    
                    if test_dir.exists():
                        # Check if directory has any images
                        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                        test_images = []
                        for ext in image_extensions:
                            test_images.extend(list(test_dir.glob(f"*{ext}")))
                            test_images.extend(list(test_dir.glob(f"*{ext.upper()}")))
                        
                        if test_images:
                            images_dir = test_dir
                            used_key = key
                            print(f"  ✓ Found {len(test_images)} images in: {images_dir}")
                            break
                        else:
                            print(f"    Directory exists but contains no images")
                    else:
                        print(f"    Directory does not exist")
                
                if images_dir:
                    break
            
            if images_dir is None:
                error_msg = "No valid image directories found. Checked:\n"
                for key, path in possible_paths:
                    error_msg += f"  {key}: {path}\n"
                    # Show what directories were actually checked
                    if path.startswith('../'):
                        dir1 = (dataset_dir / path).resolve()
                        dir2 = Path(path).resolve()
                        clean_path = path.replace('../', '')
                        dir3 = dataset_dir / clean_path
                        error_msg += f"    → {dir1}\n"
                        error_msg += f"    → {dir2}\n"
                        error_msg += f"    → {dir3}\n"
                messagebox.showerror("Error", error_msg)
                return
            
            # Find all image files in the selected directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            self.dataset_images = []
            image_files_set = set()  # Use set to avoid duplicates
            
            for ext in image_extensions:
                found_images = list(images_dir.glob(f"*{ext}"))
                found_images.extend(list(images_dir.glob(f"*{ext.upper()}")))
                # Add to set to avoid duplicates (Windows glob might be case-insensitive)
                for img in found_images:
                    image_files_set.add(img)
            
            self.dataset_images = list(image_files_set)
            
            if not self.dataset_images:
                messagebox.showerror("Error", f"No images found in {images_dir}")
                return
            
            # Sort images for consistent order
            self.dataset_images.sort()
            self.current_image_index = 0
            
            # Update UI
            self.dataset_label.config(text=f"{Path(self.dataset_path).name} ({used_key})", foreground="green")
            self.prev_button.config(state="normal")
            self.next_button.config(state="normal")
            
            # Load first image
            self.load_current_dataset_image()
            
            self.status_var.set(f"Dataset loaded: {len(self.dataset_images)} images from {used_key}")
            print(f"Successfully loaded {len(self.dataset_images)} images from {used_key} directory")
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def load_current_dataset_image(self):
        """Load current image from dataset"""
        if not self.dataset_images:
            return
        
        # Get current image path
        current_image_path = self.dataset_images[self.current_image_index]
        self.image_path = str(current_image_path)
        
        # Update image label
        self.image_label.config(text=current_image_path.name, foreground="green")
        
        # Try to find corresponding label file
        # Method 1: Same directory as image but with .txt extension
        label_path = current_image_path.with_suffix('.txt')
        
        # Method 2: Replace 'images' with 'labels' in path
        labels_path = Path(str(current_image_path).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\'))
        if labels_path.suffix != '.txt':
            labels_path = labels_path.with_suffix('.txt')
        
        # Method 3: Go up and look for labels directory
        images_parent = current_image_path.parent
        dataset_root = images_parent.parent  # Go up from images directory
        labels_dir = dataset_root / 'labels' / images_parent.name
        alt_label_path = labels_dir / current_image_path.with_suffix('.txt').name
        
        print(f"Looking for label file:")
        print(f"  Method 1: {label_path}")
        print(f"  Method 2: {labels_path}")
        print(f"  Method 3: {alt_label_path}")
        
        if label_path.exists():
            self.label_path = str(label_path)
            self.label_label.config(text=label_path.name, foreground="green")
            print(f"Found label file: {label_path}")
        elif labels_path.exists():
            self.label_path = str(labels_path)
            self.label_label.config(text=labels_path.name, foreground="green")
            print(f"Found label file: {labels_path}")
        elif alt_label_path.exists():
            self.label_path = str(alt_label_path)
            self.label_label.config(text=alt_label_path.name, foreground="green")
            print(f"Found label file: {alt_label_path}")
        else:
            self.label_path = ""
            self.label_label.config(text="No label file found", foreground="orange")
            print(f"No label file found for {current_image_path.name}")
        
        # Update navigation info
        self.image_info_label.config(text=f"{self.current_image_index + 1} / {len(self.dataset_images)}")
        
        # Update button states
        self.prev_button.config(state="normal" if self.current_image_index > 0 else "disabled")
        self.next_button.config(state="normal" if self.current_image_index < len(self.dataset_images) - 1 else "disabled")
        
        # Update run button state
        self.update_run_button_state()
    
    def previous_image(self):
        """Go to previous image in dataset"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_dataset_image()
    
    def next_image(self):
        """Go to next image in dataset"""
        if self.current_image_index < len(self.dataset_images) - 1:
            self.current_image_index += 1
            self.load_current_dataset_image()
            
    def update_run_button_state(self):
        """Enable/disable run button based on selections"""
        if self.model_path and self.image_path:
            self.run_button.config(state="normal")
        else:
            self.run_button.config(state="disabled")
            
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            # Check if model file exists
            if not Path(self.model_path).exists() and not self.model_path.startswith(('yolo', 'yolov')):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Try to load the model with better error handling
            try:
                self.model = YOLO(self.model_path)
                
                # Test if model can make a simple prediction
                test_image = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model(test_image, verbose=False)
                
            except Exception as e:
                if "C3k2" in str(e) or "attribute" in str(e):
                    # Try with a different model version
                    self.status_var.set("Trying alternative model...")
                    self.root.update()
                    
                    # Try YOLOv8 instead
                    self.model = YOLO("yolov8n.pt")
                    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
                    _ = self.model(test_image, verbose=False)
                    
                    messagebox.showwarning("Model Compatibility", 
                                         f"Original model had compatibility issues. Using YOLOv8n instead.\n"
                                         f"Original error: {str(e)}")
                else:
                    raise e
            
            self.status_var.set("Model loaded successfully")
            
            # Initialize consistent color mapping for classes after model is loaded
            self.class_colors = plt.cm.Set1(np.linspace(0, 1, len(self.class_names)))
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}\n\n"
            error_msg += "Possible solutions:\n"
            error_msg += "1. Try using 'Use Pre-trained YOLOv8n' button\n"
            error_msg += "2. Update ultralytics: pip install --upgrade ultralytics\n"
            error_msg += "3. Use a different model file\n"
            error_msg += "4. Check if the model was trained with a compatible ultralytics version"
            
            messagebox.showerror("Model Loading Error", error_msg)
            self.status_var.set("Failed to load model")
            return False
            
    def load_ground_truth(self):
        """Load ground truth labels from label file (supports both YOLO bbox and polygon formats)"""
        if not self.label_path:
            return None
            
        try:
            ground_truth = []
            with open(self.label_path, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        parts = line.split()
                        class_id = int(parts[0])
                        
                        if len(parts) == 5:
                            # Standard YOLO bounding box format: class x_center y_center width height
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            ground_truth.append({
                                'class_id': class_id,
                                'format': 'bbox',
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height
                            })
                            
                        elif len(parts) >= 9 and (len(parts) - 1) % 2 == 0:
                            # Polygon format: class x1 y1 x2 y2 x3 y3 x4 y4 ... (any number of points)
                            polygon_points = []
                            for i in range(1, len(parts), 2):
                                if i + 1 < len(parts):  # Make sure we have both x and y
                                    x = float(parts[i])
                                    y = float(parts[i+1])
                                    polygon_points.append([x, y])
                            
                            if len(polygon_points) >= 3:  # Need at least 3 points for a polygon
                                # Convert polygon to bounding box for display
                                x_coords = [p[0] for p in polygon_points]
                                y_coords = [p[1] for p in polygon_points]
                                
                                x_min, x_max = min(x_coords), max(x_coords)
                                y_min, y_max = min(y_coords), max(y_coords)
                                
                                x_center = (x_min + x_max) / 2
                                y_center = (y_min + y_max) / 2
                                width = x_max - x_min
                                height = y_max - y_min
                                
                                ground_truth.append({
                                    'class_id': class_id,
                                    'format': 'polygon',
                                    'polygon_points': polygon_points,
                                    'num_points': len(polygon_points),
                                    'x_center': x_center,
                                    'y_center': y_center,
                                    'width': width,
                                    'height': height
                                })
            
            print(f"Loaded {len(ground_truth)} ground truth annotations")
            return ground_truth
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load labels: {str(e)}")
            return None
            
    def resize_image_keep_aspect(self, image, target_size=640):
        """Resize image to target size while keeping aspect ratio"""
        h, w = image.shape[:2]
        
        # Calculate scaling factor to fit the larger dimension to target_size
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize the image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a square canvas and center the image
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Calculate positioning to center the image
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas, scale, x_offset, y_offset
        
    def run_evaluation(self):
        """Run model evaluation"""
        # Load model
        if not self.load_model():
            return
            
        # Load image
        self.status_var.set("Loading image...")
        self.root.update()
        
        original_image = cv2.imread(self.image_path)
        if original_image is None:
            messagebox.showerror("Error", f"Could not load image: {self.image_path}")
            return
            
        self.image = original_image.copy()
        
        # Load ground truth if available
        self.ground_truth = self.load_ground_truth()
        
        # Run inference
        self.status_var.set("Running inference...")
        self.root.update()
        
        # Resize image for inference
        resized_image, scale, x_offset, y_offset = self.resize_image_keep_aspect(original_image, target_size=640)
        
        # Run model prediction
        results = self.model(resized_image, conf=self.conf_var.get(), verbose=False)
        self.results = results
        
        # Store scaling info for coordinate conversion
        self.scale_info = {
            'scale': scale,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'original_size': (original_image.shape[1], original_image.shape[0])
        }
        
        # Display results
        self.display_predictions()
        if self.ground_truth:
            self.display_ground_truth()
            self.display_comparison()
        
        self.status_var.set("Evaluation completed")
        
    def display_predictions(self):
        """Display model predictions"""
        # Clear previous content
        for widget in self.pred_frame.winfo_children():
            widget.destroy()
            
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.axis('off')
        ax.set_title("Model Predictions", fontsize=16, fontweight='bold')
        
        detection_count = 0
        
        for r in self.results:
            if hasattr(r, 'obb') and r.obb is not None:
                # Process OBB (Oriented Bounding Box) results
                boxes = r.obb.xyxyxyxy.cpu().numpy()
                confidences = r.obb.conf.cpu().numpy()
                class_ids = r.obb.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    detection_count += 1
                    
                    # Scale back to original image coordinates
                    scaled_box = []
                    for point in box:
                        x = (point[0] - self.scale_info['x_offset']) / self.scale_info['scale']
                        y = (point[1] - self.scale_info['y_offset']) / self.scale_info['scale']
                        scaled_box.append([x, y])
                    
                    scaled_box = np.array(scaled_box)
                    
                    # Create polygon patch
                    colors = self.get_class_colors()
                    polygon = patches.Polygon(
                        scaled_box, 
                        linewidth=2, 
                        edgecolor=colors[class_id % len(colors)], 
                        facecolor='none',
                        alpha=0.8
                    )
                    ax.add_patch(polygon)
                    
                    # Add label
                    center_x = np.mean(scaled_box[:, 0])
                    center_y = np.mean(scaled_box[:, 1])
                    
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                    label = f"{class_name}: {conf:.2f}"
                    
                    ax.text(center_x, center_y, label, 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[class_id % len(colors)], alpha=0.7),
                           fontsize=6, ha='center', va='center')
                           
            elif hasattr(r, 'boxes') and r.boxes is not None:
                # Regular bounding boxes
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    detection_count += 1
                    
                    # Scale back to original image coordinates
                    x1 = (box[0] - self.scale_info['x_offset']) / self.scale_info['scale']
                    y1 = (box[1] - self.scale_info['y_offset']) / self.scale_info['scale']
                    x2 = (box[2] - self.scale_info['x_offset']) / self.scale_info['scale']
                    y2 = (box[3] - self.scale_info['y_offset']) / self.scale_info['scale']
                    
                    # Create rectangle patch
                    colors = self.get_class_colors()
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2, 
                        edgecolor=colors[class_id % len(colors)], 
                        facecolor='none',
                        alpha=0.8
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                    label = f"{class_name}: {conf:.2f}"
                    
                    ax.text(x1, y1-10, label, 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[class_id % len(colors)], alpha=0.7),
                           fontsize=6, ha='left', va='bottom')
        
        # Add detection count to title
        ax.set_title(f"Model Predictions - {detection_count} detections", fontsize=16, fontweight='bold')
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.pred_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        plt.close(fig)  # Close the figure to free memory
        
    def display_ground_truth(self):
        """Display ground truth labels"""
        if not self.ground_truth:
            return
            
        # Clear previous content
        for widget in self.gt_frame.winfo_children():
            widget.destroy()
            
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.axis('off')
        
        img_h, img_w = self.image.shape[:2]
        
        for i, gt in enumerate(self.ground_truth):
            class_id = gt['class_id']
            
            # Get bounding box coordinates (already converted from polygon if needed)
            x_center_norm = gt['x_center']
            y_center_norm = gt['y_center'] 
            width_norm = gt['width']
            height_norm = gt['height']
            
            # Validate normalized coordinates
            if not (0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1 and 0 <= width_norm <= 1 and 0 <= height_norm <= 1):
                continue
            
            # Convert to pixel coordinates
            x_center_px = x_center_norm * img_w
            y_center_px = y_center_norm * img_h
            width_px = width_norm * img_w
            height_px = height_norm * img_h
            
            # Calculate top-left corner coordinates
            x1 = x_center_px - width_px / 2
            y1 = y_center_px - height_px / 2
            
            # Create rectangle patch
            colors = self.get_class_colors()
            rect = patches.Rectangle(
                (x1, y1), width_px, height_px,
                linewidth=3, 
                edgecolor=colors[class_id % len(colors)], 
                facecolor='none',
                alpha=0.9,
                linestyle='--'
            )
            ax.add_patch(rect)
            
            # Add label
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
            
            ax.text(x1, y1-10, class_name, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[class_id % len(colors)], alpha=0.8),
                   fontsize=6, ha='left', va='bottom', fontweight='bold')
        
        ax.set_title(f"Ground Truth - {len(self.ground_truth)} annotations", fontsize=16, fontweight='bold')
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.gt_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        plt.close(fig)  # Close the figure to free memory
        
    def display_comparison(self):
        """Display predictions vs ground truth comparison side by side"""
        if not self.ground_truth:
            return
            
        # Clear previous content
        for widget in self.comp_frame.winfo_children():
            widget.destroy()
            
        # Create matplotlib figure with 2 subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # LEFT SUBPLOT: Model Predictions
        ax1.imshow(image_rgb)
        ax1.axis('off')
        ax1.set_title("Model Predictions", fontsize=14, fontweight='bold')
        
        detection_count = 0
        
        for r in self.results:
            if hasattr(r, 'obb') and r.obb is not None:
                # Process OBB results
                boxes = r.obb.xyxyxyxy.cpu().numpy()
                confidences = r.obb.conf.cpu().numpy()
                class_ids = r.obb.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    detection_count += 1
                    
                    # Scale back to original image coordinates
                    scaled_box = []
                    for point in box:
                        x = (point[0] - self.scale_info['x_offset']) / self.scale_info['scale']
                        y = (point[1] - self.scale_info['y_offset']) / self.scale_info['scale']
                        scaled_box.append([x, y])
                    
                    scaled_box = np.array(scaled_box)
                    
                    # Create polygon patch
                    colors = self.get_class_colors()
                    polygon = patches.Polygon(
                        scaled_box, 
                        linewidth=2, 
                        edgecolor=colors[class_id % len(colors)], 
                        facecolor='none',
                        alpha=0.8
                    )
                    ax1.add_patch(polygon)
                    
                    # Add label with class name and confidence
                    center_x = np.mean(scaled_box[:, 0])
                    center_y = np.mean(scaled_box[:, 1])
                    
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                    label = f"{class_name}: {conf:.2f}"
                    
                    ax1.text(center_x, center_y, label, 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[class_id % len(colors)], alpha=0.7),
                           fontsize=6, ha='center', va='center')
                    
            elif hasattr(r, 'boxes') and r.boxes is not None:
                # Regular bounding boxes
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    detection_count += 1
                    
                    # Scale back to original image coordinates
                    x1 = (box[0] - self.scale_info['x_offset']) / self.scale_info['scale']
                    y1 = (box[1] - self.scale_info['y_offset']) / self.scale_info['scale']
                    x2 = (box[2] - self.scale_info['x_offset']) / self.scale_info['scale']
                    y2 = (box[3] - self.scale_info['y_offset']) / self.scale_info['scale']
                    
                    # Create rectangle patch
                    colors = self.get_class_colors()
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2, 
                        edgecolor=colors[class_id % len(colors)], 
                        facecolor='none',
                        alpha=0.8
                    )
                    ax1.add_patch(rect)
                    
                    # Add label with class name and confidence
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                    label = f"{class_name}: {conf:.2f}"
                    
                    ax1.text(x1, y1-10, label, 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[class_id % len(colors)], alpha=0.7),
                           fontsize=6, ha='left', va='bottom')
        
        # Update title with detection count
        ax1.set_title(f"Model Predictions ({detection_count} detections)", fontsize=14, fontweight='bold')
        
        # RIGHT SUBPLOT: Ground Truth
        ax2.imshow(image_rgb)
        ax2.axis('off')
        ax2.set_title(f"Ground Truth ({len(self.ground_truth)} annotations)", fontsize=14, fontweight='bold')
        
        img_h, img_w = self.image.shape[:2]
        
        for gt in self.ground_truth:
            class_id = gt['class_id']
            
            # Get bounding box coordinates (already converted from polygon if needed)
            x_center_norm = gt['x_center']
            y_center_norm = gt['y_center'] 
            width_norm = gt['width']
            height_norm = gt['height']
            
            # Skip invalid coordinates
            if not (0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1 and 0 <= width_norm <= 1 and 0 <= height_norm <= 1):
                continue
            
            # Convert to pixel coordinates
            x_center_px = x_center_norm * img_w
            y_center_px = y_center_norm * img_h
            width_px = width_norm * img_w
            height_px = height_norm * img_h
            
            # Calculate top-left corner coordinates
            x1 = x_center_px - width_px / 2
            y1 = y_center_px - height_px / 2
            
            # Create rectangle patch
            colors = self.get_class_colors()
            rect = patches.Rectangle(
                (x1, y1), width_px, height_px,
                linewidth=3, 
                edgecolor=colors[class_id % len(colors)], 
                facecolor='none',
                alpha=0.9,
                linestyle='--'
            )
            ax2.add_patch(rect)
            
            # Add label with class name
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
            
            ax2.text(x1, y1-10, class_name, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[class_id % len(colors)], alpha=0.8),
                   fontsize=6, ha='left', va='bottom', fontweight='bold')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.comp_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        plt.close(fig)  # Close the figure to free memory

def main():
    root = tk.Tk()
    app = ModelEvaluationInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()
