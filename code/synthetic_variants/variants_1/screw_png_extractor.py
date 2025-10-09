#!/usr/bin/env python3
"""
Robust Screw PNG Extractor
==========================
Extracts individual screws from labeled images with transparent backgrounds.
Uses YOLO bounding box annotations to precisely crop screws from images.

Features:
- Transparent background extraction using advanced segmentation
- Maintains actual pixel dimensions
- Creates proper class-based naming
- Supports batch processing with user-defined sample size
- Multiple extraction methods (bbox, grabcut, watershed)
"""

import os
import cv2
import numpy as np
import random
from datetime import datetime
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Optional
import json
from scipy import ndimage

# Advanced segmentation libraries
from skimage import segmentation, morphology, filters, feature, measure
from skimage.color import rgb2gray
from rembg import remove, new_session
import warnings
warnings.filterwarnings('ignore')

class ScrewExtractor:
    def __init__(self, dataset_path: str, output_path: str = "extracted_screws"):
        """
        Initialize the Screw Extractor
        
        Args:
            dataset_path: Path to the YOLO dataset folder
            output_path: Path where extracted screws will be saved
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.class_names = [
            'M4-12mm', 'M4-16mm', 'M4-20mm', 'M4-8mm', 'M4-Nut', 'M4-Washer',
            'M6-12mm', 'M6-16mm', 'M6-20mm', 'M6-8mm', 'M6-Nut', 'M6-Washer',
            'Standing-Nut', 'Standing-Screw'
        ]
        
        # Create output directories
        self.setup_output_directories()
        
        # Statistics
        self.extraction_stats = {class_name: 0 for class_name in self.class_names}
        self.failed_extractions = []
        
    def setup_output_directories(self):
        """Create organized output directory structure"""
        self.output_path.mkdir(exist_ok=True)
        
        # Create class-specific directories for images and labels
        for class_name in self.class_names:
            (self.output_path / class_name).mkdir(exist_ok=True)
            (self.output_path / class_name / "images").mkdir(exist_ok=True)
            (self.output_path / class_name / "labels").mkdir(exist_ok=True)
        
        # Create metadata directory
        (self.output_path / "_metadata").mkdir(exist_ok=True)
        
        print(f"📁 Output directories created in: {self.output_path}")
        
    def load_yolo_annotations(self, label_file: Path) -> List[Dict]:
        """
        Load YOLO format annotations from label file
        
        Args:
            label_file: Path to .txt label file
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        if not label_file.exists():
            return annotations
            
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Standard YOLO bbox format
                if len(parts) == 5:
                    class_id, center_x, center_y, width, height = map(float, parts)
                    annotations.append({
                        'class_id': int(class_id),
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height,
                        'type': 'bbox'
                    })
                # OBB (Oriented Bounding Box) format
                elif len(parts) == 9:
                    class_id = int(float(parts[0]))
                    points = list(map(float, parts[1:]))
                    # Convert to 4 corner points
                    x_coords = points[::2]
                    y_coords = points[1::2]
                    annotations.append({
                        'class_id': class_id,
                        'points': list(zip(x_coords, y_coords)),
                        'type': 'obb'
                    })
                    
        return annotations
    
    def yolo_to_pixel_coords(self, annotation: Dict, img_height: int, img_width: int) -> Tuple:
        """
        Convert YOLO normalized coordinates to pixel coordinates
        
        Args:
            annotation: YOLO annotation dictionary
            img_height: Image height in pixels
            img_width: Image width in pixels
            
        Returns:
            Pixel coordinates (depends on annotation type)
        """
        if annotation['type'] == 'bbox':
            center_x = annotation['center_x'] * img_width
            center_y = annotation['center_y'] * img_height
            width = annotation['width'] * img_width
            height = annotation['height'] * img_height
            
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)
            
            return (x1, y1, x2, y2)
            
        elif annotation['type'] == 'obb':
            pixel_points = []
            for x, y in annotation['points']:
                pixel_x = int(x * img_width)
                pixel_y = int(y * img_height)
                pixel_points.append((pixel_x, pixel_y))
            return pixel_points
    
    def extract_with_bbox_method(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                                padding: int = 10) -> np.ndarray:
        """
        Simple bounding box extraction with padding
        
        Args:
            image: Input image
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            padding: Padding around bounding box
            
        Returns:
            Extracted image region
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Apply padding with bounds checking
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return image[y1:y2, x1:x2]
    
    def extract_with_grabcut(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Advanced extraction using GrabCut algorithm for better background removal
        
        Args:
            image: Input image
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            Extracted image with transparent background
        """
        x1, y1, x2, y2 = bbox
        
        # Create mask
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Background and foreground models for GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Define rectangle for GrabCut
        rect = (x1, y1, x2-x1, y2-y1)
        
        try:
            # Apply GrabCut
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create final mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Extract the region
            result = image * mask2[:, :, np.newaxis]
            
            # Crop to bounding box
            cropped = result[y1:y2, x1:x2]
            
            # Create alpha channel
            alpha = mask2[y1:y2, x1:x2] * 255
            
            # Combine RGB with alpha
            if cropped.shape[2] == 3:
                result_with_alpha = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
                result_with_alpha[:, :, 3] = alpha
                return result_with_alpha
                
        except Exception as e:
            print(f"⚠️  GrabCut failed: {e}, falling back to bbox method")
            
        # Fallback to simple bbox extraction
        return self.extract_with_bbox_method(image, bbox)
    
    def extract_with_obb(self, image: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Extract using oriented bounding box (polygon)
        
        Args:
            image: Input image
            points: List of corner points
            
        Returns:
            Extracted and rotated image
        """
        # Convert points to numpy array
        points_array = np.array(points, dtype=np.int32)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(points_array)
        
        # Create mask for the polygon
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points_array], 255)
        
        # Extract the region
        result = cv2.bitwise_and(image, image, mask=mask)
        
        # Crop to bounding rectangle
        cropped = result[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        
        # Create transparent background
        if cropped.shape[2] == 3:
            result_with_alpha = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
            result_with_alpha[:, :, 3] = cropped_mask
            return result_with_alpha
        
        return cropped
    
    def extract_with_advanced_segmentation(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced segmentation that removes background INSIDE the bounding box
        
        Args:
            image: Input image
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            Tuple of (extracted_image_with_alpha, binary_mask_for_original_bbox)
        """
        x1, y1, x2, y2 = bbox
        
        # First extract the exact bounding box region
        cropped_region = image[y1:y2, x1:x2]
        
        if cropped_region.size == 0:
            return None, None
        
        # Try segmentation directly on the cropped region
        print(f"🔬 Starting advanced segmentation on {cropped_region.shape} region...")
        mask = self._get_best_segmentation_mask(cropped_region)
        
        if mask is None:
            print(f"⚠️  Segmentation failed, using fallback")
            # Fallback: create a mask that removes border pixels only
            mask = self._create_border_removal_mask(cropped_region)
        
        # Apply mask with transparency
        result_with_alpha = self._apply_mask_with_transparency(cropped_region, mask)
        
        # Debug: Print mask statistics
        fg_pixels = np.sum(mask == 255)
        total_pixels = mask.shape[0] * mask.shape[1]
        bg_removed_ratio = 1.0 - (fg_pixels / total_pixels)
        print(f"🎭 Background removed: {bg_removed_ratio:.1%} ({total_pixels - fg_pixels}/{total_pixels} pixels)")
        
        return result_with_alpha, mask
    
    def advanced_background_removal(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Apply advanced background removal techniques with CONSERVATIVE approach
        Prioritizes preserving screw integrity over aggressive background removal
        Returns: (image_with_alpha, background_removal_percentage)
        """
        if image.size == 0:
            return None, 0.0
        
        print(f"🔬 Starting CONSERVATIVE background removal on {image.shape} image...")
        
        # Try multiple methods but be more selective about what we accept
        methods = [
            ('rembg_ai', self._rembg_background_removal),
            ('superpixel_slic', self._superpixel_segmentation_advanced),
            ('adaptive_grabcut_advanced', self._adaptive_grabcut_advanced),
            ('watershed_advanced', self._watershed_segmentation_advanced),
            ('color_clustering', self._color_clustering_segmentation),
            ('edge_detection', self._edge_detection_segmentation),
            ('threshold_otsu', self._otsu_segmentation),
            ('conservative_threshold', self._conservative_threshold_segmentation),
            ('gentle_superpixel', self._gentle_superpixel_segmentation),
            ('conservative_grabcut', self._conservative_grabcut),
            ('edge_aware', self._edge_aware_segmentation),
        ]
        
        best_result = None
        best_score = 0
        best_bg_removal = 0
        
        for method_name, method_func in methods:
            try:
                mask = method_func(image)
                if mask is not None:
                    score = self._evaluate_mask_quality(mask, image)
                    bg_removal_pct = self._calculate_background_removal(mask)
                    fg_ratio = np.sum(mask == 255) / mask.size
                    
                    print(f"  🧪 {method_name}: score={score:.3f}, bg_removal={bg_removal_pct:.1f}%, fg_ratio={fg_ratio:.1%}")
                    
                    # MUCH more conservative acceptance criteria
                    # Only accept if we preserve 40-90% of the image (avoid over-removal)
                    if score > best_score and 0.4 <= fg_ratio <= 0.9:
                        best_score = score
                        best_result = mask
                        best_bg_removal = bg_removal_pct
                        print(f"   ✅ New best method: {method_name}")
                        
            except Exception as e:
                print(f"   ❌ {method_name} failed: {e}")
                continue
        
        # Apply best mask OR skip this screw if quality is insufficient
        if best_result is not None and best_score > 0.5:  # Higher threshold
            result_image = self._apply_mask_with_transparency(image, best_result)
            print(f"✅ Applied segmentation: {best_bg_removal:.1f}% background removed")
            return result_image, best_bg_removal
        else:
            print("🔄 Segmentation quality insufficient, skipping this screw")
            return None, 0.0
    
    def _rembg_background_removal(self, image: np.ndarray) -> Optional[np.ndarray]:
        """AI-powered background removal using rembg"""
        try:
            # Convert BGR to RGB for rembg
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply rembg
            result = remove(rgb_image)
            
            # Extract alpha channel as mask
            if result.shape[2] == 4:
                mask = result[:, :, 3]
                print(f"🤖 AI background removal applied")
                return mask
            
        except Exception as e:
            print(f"❌ rembg failed: {e}")
        
        return None
    
    def _superpixel_segmentation_advanced(self, image: np.ndarray) -> Optional[np.ndarray]:
        """SLIC superpixel-based segmentation"""
        try:
            # Apply SLIC superpixels
            segments = segmentation.slic(image, n_segments=100, compactness=10, 
                                       sigma=1, start_label=1)
            
            # Find central segments (likely to be object)
            h, w = segments.shape
            center_y, center_x = h // 2, w // 2
            center_segment = segments[center_y, center_x]
            
            # Create mask for central segments and neighbors
            mask = np.zeros(segments.shape, dtype=np.uint8)
            
            # Include center segment and similar segments
            for segment_id in np.unique(segments):
                segment_mask = segments == segment_id
                segment_center = np.mean(np.where(segment_mask), axis=1)
                
                # Distance from image center
                dist = np.sqrt((segment_center[0] - center_y)**2 + (segment_center[1] - center_x)**2)
                
                # Include if close to center
                if dist < min(h, w) * 0.3:  # Within 30% of image center
                    mask[segment_mask] = 255
            
            # Morphological cleanup
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            print(f"🔬 SLIC superpixel segmentation applied")
            return mask
            
        except Exception as e:
            print(f"❌ Superpixel segmentation failed: {e}")
        
        return None
    
    def _adaptive_grabcut_advanced(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Adaptive GrabCut segmentation - advanced version"""
        try:
            h, w = image.shape[:2]
            if h < 50 or w < 50:
                return None
            
            # Initialize mask and models
            mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Define rectangle (leave 15% border)
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)
            rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
            
            if rect[2] <= 0 or rect[3] <= 0:
                return None
            
            # Apply GrabCut
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask
            final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Morphological cleanup
            kernel = np.ones((3, 3), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            print(f"✂️ Advanced Adaptive GrabCut applied")
            return final_mask * 255
            
        except Exception as e:
            print(f"❌ Advanced GrabCut failed: {e}")
        
        return None
    
    def _watershed_segmentation_advanced(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Advanced watershed segmentation"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Apply watershed
            markers = cv2.watershed(image, markers)
            
            # Create final mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[markers > 1] = 255
            
            print(f"💧 Advanced watershed applied")
            return mask
            
        except Exception as e:
            print(f"❌ Advanced watershed failed: {e}")
        
        return None
    
    def _edge_detection_segmentation(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Edge detection based segmentation"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise while keeping edges sharp
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Edge detection
            edges = cv2.Canny(filtered, 50, 150)
            
            # Close edges to form contours
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.erode(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find the largest contour (likely the main object)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            print(f"🔍 Edge detection applied")
            return mask
            
        except Exception as e:
            print(f"❌ Edge detection failed: {e}")
        
        return None
    
    def _otsu_segmentation(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Otsu threshold-based segmentation"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Otsu thresholding
            _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            print(f"📊 Otsu segmentation applied")
            return mask
            
        except Exception as e:
            print(f"❌ Otsu segmentation failed: {e}")
        
        return None
    
    def _create_transparent_image(self, image: np.ndarray) -> np.ndarray:
        """Create transparent version of image (fallback)"""
        result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        result[:, :, :3] = image
        result[:, :, 3] = 255  # Fully opaque
        return result
    
    def _conservative_threshold_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Very conservative thresholding that preserves screw details"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Gentle bilateral filter
        filtered = cv2.bilateralFilter(gray, 5, 30, 30)
        
        # Use adaptive thresholding instead of Otsu (more conservative)
        mask = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, 3)
        
        # Very gentle morphological operations
        kernel = np.ones((2,2), np.uint8)
        
        # Only close small holes, avoid opening (which removes object parts)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return mask
    
    def _gentle_superpixel_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Gentle superpixel approach that keeps more of the object"""
        try:
            # Use larger superpixels for gentler segmentation
            segments = segmentation.slic(roi, n_segments=30, compactness=15, 
                                       sigma=2, start_label=1)
            
            # Find all segments that are NOT purely on edges
            h, w = segments.shape
            edge_segments = set()
            
            # Mark edge segments (first/last 2 rows and columns)
            edge_segments.update(segments[0:2, :].flatten())
            edge_segments.update(segments[-2:, :].flatten())  
            edge_segments.update(segments[:, 0:2].flatten())
            edge_segments.update(segments[:, -2:].flatten())
            
            # Create mask - keep segments that are NOT purely edges
            mask = np.ones((h, w), dtype=np.uint8) * 255
            for segment_id in edge_segments:
                segment_mask = segments == segment_id
                # Only remove if segment is >80% on edge
                edge_coverage = np.sum(segment_mask) / (h * w)
                if edge_coverage > 0.1:  # Don't remove large segments
                    continue
                mask[segment_mask] = 0
            
            return mask
            
        except Exception as e:
            print(f"⚠️ Gentle superpixel failed: {e}")
            return self._conservative_threshold_segmentation(roi)
    
    def _conservative_grabcut(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Very conservative GrabCut that preserves object"""
        try:
            h, w = image.shape[:2]
            if h < 30 or w < 30:
                return None
            
            mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Use smaller border (only 5% instead of 15%)
            margin_x = max(2, int(w * 0.05))
            margin_y = max(2, int(h * 0.05))
            rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
            
            if rect[2] <= 0 or rect[3] <= 0:
                return None
            
            # Apply GrabCut with fewer iterations for gentler result
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask - be more inclusive
            final_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
            
            # Only gentle cleanup
            kernel = np.ones((2, 2), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            return final_mask
            
        except Exception as e:
            print(f"❌ Conservative GrabCut failed: {e}")
        
        return None
    
    def _edge_aware_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Edge-aware segmentation that preserves object boundaries"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Gentle edge detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100)  # Lower thresholds for gentler edges
        
        # Very gentle dilation
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours and keep larger ones
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback to conservative threshold
            return self._conservative_threshold_segmentation(roi)
        
        # Create mask and fill larger contours
        mask = np.ones(gray.shape, dtype=np.uint8) * 255  # Start with white (keep everything)
        total_area = gray.shape[0] * gray.shape[1]
        
        # Only remove very small contours (likely noise)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < total_area * 0.002:  # Only remove tiny areas (<0.2% of image)
                cv2.fillPoly(mask, [contour], 0)
        
        return mask
    
    def _adaptive_grabcut(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Adaptive GrabCut segmentation"""
        try:
            h, w = image.shape[:2]
            if h < 50 or w < 50:
                return None
            
            # Initialize mask and models
            mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Define rectangle (leave 15% border)
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)
            rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
            
            if rect[2] <= 0 or rect[3] <= 0:
                return None
            
            # Apply GrabCut
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask
            final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Morphological cleanup
            kernel = np.ones((3, 3), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            print(f"✂️ Adaptive GrabCut applied")
            return final_mask * 255
            
        except Exception as e:
            print(f"❌ GrabCut failed: {e}")
        
        return None
    
    def _calculate_background_removal(self, mask: np.ndarray) -> float:
        """Calculate percentage of background removed"""
        if mask is None or mask.size == 0:
            return 0.0
        
        fg_pixels = np.sum(mask == 255)
        total_pixels = mask.size
        bg_removal_pct = (1 - fg_pixels / total_pixels) * 100
        
        return bg_removal_pct
    
    def _create_conservative_mask(self, image: np.ndarray) -> np.ndarray:
        """Create conservative elliptical mask (fallback)"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create elliptical mask (conservative - keeps most of the object)
        center = (w // 2, h // 2)
        axes = (int(w * 0.4), int(h * 0.4))  # 40% of dimensions
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        result = self._apply_mask_with_transparency(image, mask)
        print("🛡️ Conservative elliptical mask applied")
        
        return result
    
    def _get_best_segmentation_mask(self, roi: np.ndarray) -> np.ndarray:
        """
        Try multiple segmentation methods and return the best mask
        NEW: Much more aggressive background removal
        """
        methods = [
            ('rembg_ai', self._rembg_segmentation),  # AI-powered - try first
            ('superpixel', self._superpixel_segmentation),  # Advanced superpixel method
            ('multiscale', self._multiscale_segmentation),  # Consensus of multiple methods
            ('advanced_watershed', self._advanced_watershed_segmentation),  # Enhanced watershed
            ('canny_edge_detection', self._canny_edge_segmentation),  # Enhanced edge detection
            ('canny_grabcut_hybrid', self._canny_grabcut_hybrid_segmentation),  # Enhanced Edge + GrabCut
            ('metal_object_segmentation', self._metal_object_segmentation),  # NEW: Metal-specific segmentation
            ('aggressive_threshold', self._aggressive_threshold_segmentation),
            ('edge_fill', self._edge_fill_segmentation),
            ('color_clustering', self._color_clustering_segmentation),
            ('adaptive_threshold', self._adaptive_threshold_segmentation),
        ]
        
        # NEW: Use more balanced approach - prefer moderate background removal over losing screw parts
        best_mask = None
        best_score = 0
        
        print(f"🔍 Trying {len(methods)} advanced segmentation methods...")
        
        for method_name, method_func in methods:
            try:
                print(f"  🧪 Testing {method_name}...")
                mask = method_func(roi)
                if mask is not None:
                    fg_ratio = np.sum(mask == 255) / mask.size
                    score = self._evaluate_mask_quality(mask, roi)
                    print(f"   {method_name}: score={score:.3f}, fg_ratio={fg_ratio:.1%}")
                    
                    # NEW: Prefer methods with good quality score, avoid over-aggressive removal
                    if score > best_score and 0.2 <= fg_ratio <= 0.8:  # Keep reasonable foreground
                        best_score = score
                        best_mask = mask
                        print(f"   ✅ New best method: {method_name} (score={score:.3f}, fg={fg_ratio:.1%})")
            except Exception as e:
                print(f"   ❌ {method_name} failed: {e}")
                continue
        
        # Less aggressive fallback - only use if no good method found
        if best_mask is None:
            print(f"⚡ No good segmentation found, using conservative fallback")
            best_mask = self._conservative_fallback_mask(roi)
        else:
            final_fg_ratio = np.sum(best_mask == 255) / best_mask.size
            print(f"✅ Selected method preserves {final_fg_ratio:.1%} of screw")
        
        if best_mask is not None:
            fg_pixels = np.sum(best_mask == 255)
            total_pixels = best_mask.shape[0] * best_mask.shape[1]
            bg_removed_ratio = 1.0 - (fg_pixels / total_pixels)
            print(f"🎭 Final result: {bg_removed_ratio:.1%} background removed ({total_pixels - fg_pixels}/{total_pixels} pixels)")
        
        return best_mask if best_mask is not None else self._conservative_fallback_mask(roi)
    
    def _aggressive_threshold_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Balanced thresholding - removes background while preserving screw details"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply moderate bilateral filter
        filtered = cv2.bilateralFilter(gray, 9, 50, 50)
        
        # Use Otsu thresholding (often good balance)
        _, otsu_mask = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # More conservative morphological operations
        kernel = np.ones((3,3), np.uint8)
        
        # Remove small noise
        mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill small holes in objects (preserve screw integrity)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask
    
    def _edge_fill_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Edge-based segmentation - balanced approach"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Moderate blur to reduce noise while preserving edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        
        # Edge detection with balanced thresholds
        edges = cv2.Canny(blurred, 50, 150)
        
        # Moderate dilation to close gaps
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Create mask and fill the largest contour(s)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Fill multiple large contours (in case object is split)
        contours_by_area = sorted(contours, key=cv2.contourArea, reverse=True)
        total_area = gray.shape[0] * gray.shape[1]
        
        for contour in contours_by_area[:2]:  # Top 2 contours only
            area = cv2.contourArea(contour)
            if area > total_area * 0.03:  # At least 3% of image (more permissive)
                cv2.fillPoly(mask, [contour], 255)
        
        # Gentle erosion to clean edges without cutting screw parts
        small_kernel = np.ones((2,2), np.uint8)
        mask = cv2.erode(mask, small_kernel, iterations=1)
        
        return mask
    
    def _color_clustering_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Color-based segmentation using simple clustering"""
        # Convert to LAB color space for better color separation
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # Reshape for clustering
        pixels = lab.reshape(-1, 3).astype(np.float32)
        
        # Simple K-means clustering (k=2: object vs background)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Reshape back
        labels = labels.reshape(roi.shape[:2])
        
        # Determine which cluster is foreground (assume it's the smaller one or the one not touching edges)
        cluster0_size = np.sum(labels == 0)
        cluster1_size = np.sum(labels == 1)
        
        # Check which cluster dominates the edges (more likely to be background)
        h, w = labels.shape
        edge_pixels = np.concatenate([
            labels[0, :],     # top edge
            labels[-1, :],    # bottom edge  
            labels[:, 0],     # left edge
            labels[:, -1]     # right edge
        ])
        
        cluster0_edge_count = np.sum(edge_pixels == 0)
        cluster1_edge_count = np.sum(edge_pixels == 1)
        
        # Foreground is the cluster with fewer edge pixels
        if cluster0_edge_count < cluster1_edge_count:
            foreground_label = 0
        else:
            foreground_label = 1
        
        mask = (labels == foreground_label).astype(np.uint8) * 255
        
        # Gentle cleanup to preserve screw details
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
    
    def _conservative_fallback_mask(self, roi: np.ndarray) -> np.ndarray:
        """Conservative fallback: Preserve screw shape, remove only obvious background"""
        mask = np.ones(roi.shape[:2], dtype=np.uint8) * 255
        h, w = roi.shape[:2]
        
        print(f"🛡️ Using conservative fallback mask for {w}x{h} region")
        
        # Only remove corners and obvious background areas
        corner_size = min(h, w) // 8  # Remove small corners only
        
        # Remove corner triangles (typical background areas)
        for y in range(corner_size):
            for x in range(corner_size):
                if x + y < corner_size:
                    mask[y, x] = 0  # Top-left corner
                if (w - x - 1) + y < corner_size:
                    mask[y, w - x - 1] = 0  # Top-right corner
                if x + (h - y - 1) < corner_size:
                    mask[h - y - 1, x] = 0  # Bottom-left corner
                if (w - x - 1) + (h - y - 1) < corner_size:
                    mask[h - y - 1, w - x - 1] = 0  # Bottom-right corner
        
        return mask
    
    def _fallback_aggressive_mask(self, roi: np.ndarray) -> np.ndarray:
        """Fallback: ULTRA aggressive elliptical mask - removes most background"""
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        h, w = roi.shape[:2]
        
        print(f"⚡ Using ultra-aggressive fallback mask for {w}x{h} region")
        
        # Create VERY aggressive elliptical mask (only keep center 40%)
        center_x, center_y = w // 2, h // 2
        
        # Make ellipse much smaller - only keep the core of the object
        axis_a = max(10, int(w * 0.25))  # 25% of width, minimum 10 pixels
        axis_b = max(10, int(h * 0.25))  # 25% of height, minimum 10 pixels
        
        cv2.ellipse(mask, (center_x, center_y), (axis_a, axis_b), 0, 0, 360, 255, -1)
        
        # Additional erosion to make it even smaller
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        return mask
    
    def _watershed_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Watershed segmentation for metal objects"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Use Otsu thresholding to separate foreground and background
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise with morphological operations
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area (dilate a lot)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Sure foreground area using distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # Be more aggressive - use lower threshold to capture more of the object
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        
        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(roi, markers)
        
        # Create mask - be more inclusive of the object
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers > 1] = 255
        
        # Post-process to fill holes and smooth
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask
    
    def _adaptive_threshold_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Adaptive thresholding segmentation"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold
        mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _edge_based_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Edge-based segmentation using Canny and contour detection"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to close gaps
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask from contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [largest_contour], 255)
        
        return mask
    
    def _grabcut_roi_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """GrabCut segmentation on ROI"""
        if roi.shape[0] < 50 or roi.shape[1] < 50:
            return None
        
        mask = np.zeros(roi.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Define rectangle (leave border)
        height, width = roi.shape[:2]
        rect = (10, 10, width-20, height-20)
        
        if rect[2] <= 0 or rect[3] <= 0:
            return None
        
        cv2.grabCut(roi, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        
        # Create binary mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Post-process to clean up
        kernel = np.ones((3,3), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        
        return mask2 * 255
    
    def _create_border_removal_mask(self, roi: np.ndarray) -> np.ndarray:
        """Create a mask that removes border pixels (fallback method)"""
        mask = np.ones(roi.shape[:2], dtype=np.uint8) * 255
        h, w = roi.shape[:2]
        
        # Remove border pixels (create oval/elliptical mask)
        center_x, center_y = w // 2, h // 2
        for y in range(h):
            for x in range(w):
                # Calculate distance from center, normalized by image dimensions
                dx = abs(x - center_x) / (w / 2)
                dy = abs(y - center_y) / (h / 2)
                
                # Create elliptical mask (remove pixels far from center)
                if (dx * dx + dy * dy) > 0.8:  # Adjust threshold as needed
                    mask[y, x] = 0
        
        return mask
    
    def _superpixel_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Advanced superpixel-based segmentation using SLIC"""
        try:
            # Convert to float for skimage
            roi_float = roi.astype(np.float32) / 255.0
            
            # Apply SLIC superpixel segmentation
            segments = segmentation.slic(roi_float, n_segments=80, compactness=10, sigma=1, start_label=1)
            
            # Analyze segments to identify foreground vs background
            h, w = segments.shape
            edge_segments = set()
            
            # Collect edge segments (likely background)
            edge_segments.update(segments[0, :])  # Top edge
            edge_segments.update(segments[-1, :])  # Bottom edge  
            edge_segments.update(segments[:, 0])  # Left edge
            edge_segments.update(segments[:, -1])  # Right edge
            
            # Create mask - segments not on edges are likely foreground
            mask = np.zeros((h, w), dtype=np.uint8)
            for segment_id in np.unique(segments):
                if segment_id not in edge_segments:
                    mask[segments == segment_id] = 255
            
            # Clean up the mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            print(f"🔬 Applied SLIC superpixel segmentation ({np.sum(mask==255)/mask.size*100:.1f}% foreground)")
            return mask
            
        except Exception as e:
            print(f"⚠️ Superpixel segmentation failed: {e}")
            return self._conservative_fallback_mask(roi)
    
    def _rembg_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """AI-powered background removal using rembg"""
        try:
            # Convert to PIL format (RGB)
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Apply rembg background removal
            output = remove(roi_rgb)
            
            # Extract mask from alpha channel
            if len(output.shape) == 3 and output.shape[2] == 4:  # RGBA
                mask = output[:, :, 3]  # Alpha channel
                # Convert to binary mask
                mask = (mask > 127).astype(np.uint8) * 255
            else:
                # Fallback: convert to grayscale and threshold
                if len(output.shape) == 3:
                    gray = rgb2gray(output)
                else:
                    gray = output
                mask = (gray > 0.1).astype(np.uint8) * 255
            
            # Clean up
            kernel = np.ones((2,2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            print(f"🤖 Applied AI background removal ({np.sum(mask==255)/mask.size*100:.1f}% foreground)")
            return mask
            
        except Exception as e:
            print(f"⚠️ AI background removal failed: {e}")
            return self._aggressive_threshold_segmentation(roi)
    
    def _multiscale_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Multi-scale segmentation combining different techniques"""
        # Get multiple segmentation results
        masks = []
        
        # Threshold-based
        thresh_mask = self._aggressive_threshold_segmentation(roi)
        if thresh_mask is not None:
            masks.append(thresh_mask.astype(np.float32) / 255.0)
        
        # Edge-based
        edge_mask = self._edge_fill_segmentation(roi)
        if edge_mask is not None:
            masks.append(edge_mask.astype(np.float32) / 255.0)
            
        # Color clustering
        color_mask = self._color_clustering_segmentation(roi)
        if color_mask is not None:
            masks.append(color_mask.astype(np.float32) / 255.0)
        
        if not masks:
            return self._conservative_fallback_mask(roi)
        
        # Combine masks using majority voting
        combined = np.zeros_like(masks[0])
        for mask in masks:
            combined += mask
        
        # Threshold: areas where majority of methods agree (60% consensus)
        final_mask = (combined >= (len(masks) * 0.6)).astype(np.uint8) * 255
        
        # Clean up
        kernel = np.ones((3,3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        print(f"🔄 Applied multi-scale consensus ({np.sum(final_mask==255)/final_mask.size*100:.1f}% foreground)")
        return final_mask
    
    def _advanced_watershed_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Advanced watershed with distance transform and better markers"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Enhanced preprocessing
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Multiple thresholding approaches
            _, otsu_mask = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Distance transform for better marker detection
            dist_transform = cv2.distanceTransform(otsu_mask, cv2.DIST_L2, 5)
            
            # Find peaks as markers
            _, markers = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
            markers = markers.astype(np.uint8)
            
            # Connected components for markers
            _, markers = cv2.connectedComponents(markers)
            
            # Apply watershed
            markers = cv2.watershed(roi, markers)
            
            # Create final mask
            mask = (markers > 1).astype(np.uint8) * 255
            
            # Post-process
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            print(f"💧 Applied advanced watershed ({np.sum(mask==255)/mask.size*100:.1f}% foreground)")
            return mask
            
        except Exception as e:
            print(f"⚠️ Advanced watershed failed: {e}")
            return self._watershed_segmentation(roi)
    
    def _canny_edge_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Enhanced edge detection using Canny with better preprocessing"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Enhanced preprocessing for metal objects
            # 1. Bilateral filter to preserve edges while reducing noise
            filtered = cv2.bilateralFilter(gray, 9, 80, 80)
            
            # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(filtered)
            
            # 3. Multiple edge detection with different parameters
            edges1 = cv2.Canny(enhanced, 30, 100)  # More sensitive
            edges2 = cv2.Canny(enhanced, 50, 150)  # Medium
            edges3 = cv2.Canny(enhanced, 70, 200)  # Less sensitive
            
            # Combine edges using OR operation
            combined_edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
            
            # Morphological operations to close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            closed_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find and filter contours by area
            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Filter contours by area - keep larger ones
            total_area = roi.shape[0] * roi.shape[1]
            filtered_contours = [c for c in contours if cv2.contourArea(c) > total_area * 0.01]
            
            if not filtered_contours:
                filtered_contours = contours  # Use all if none pass filter
            
            # Create mask and fill contours
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
            
            # Advanced post-processing
            # 1. Fill holes inside objects
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            
            # 2. Remove small noise
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
            
            # 3. Smooth the edges
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            print(f"🔍 Applied Enhanced Canny edge detection ({np.sum(mask==255)/mask.size*100:.1f}% foreground)")
            return mask
            
        except Exception as e:
            print(f"⚠️ Enhanced Canny edge detection failed: {e}")
            return None
    
    def _canny_grabcut_hybrid_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Enhanced hybrid: Better edge detection + GrabCut with improved initialization"""
        try:
            # Step 1: Enhanced preprocessing
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Bilateral filter to preserve edges
            filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Step 2: Multi-scale edge detection
            edges1 = cv2.Canny(filtered, 20, 80)   # Very sensitive for fine details
            edges2 = cv2.Canny(filtered, 40, 120)  # Medium sensitivity
            edges3 = cv2.Canny(filtered, 60, 180)  # Less sensitive for main structure
            
            # Combine all edge maps
            combined_edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
            
            # Close gaps in edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Step 3: Create better initial mask for GrabCut
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Filter contours and create initial mask
            total_area = roi.shape[0] * roi.shape[1]
            mask_init = np.zeros_like(gray)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > total_area * 0.005:  # Keep contours > 0.5% of image area
                    cv2.fillPoly(mask_init, [contour], 255)
            
            # Dilate the initial mask slightly to ensure we capture the object
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_init = cv2.dilate(mask_init, kernel_dilate, iterations=1)
            
            # Step 4: Use enhanced mask for GrabCut initialization
            # Create proper GrabCut mask: probable foreground in edge areas
            mask_gc = np.zeros_like(gray, dtype=np.uint8)
            
            # Areas inside the edge mask are probably foreground
            mask_gc[mask_init == 255] = cv2.GC_PR_FGD
            
            # Create a border around the mask as probable background
            border_mask = cv2.dilate(mask_init, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), iterations=1)
            mask_gc[border_mask == 255] = cv2.GC_PR_BGD
            mask_gc[mask_init == 255] = cv2.GC_PR_FGD  # Restore foreground areas
            
            # Definitely background (corners and edges of image)
            h, w = gray.shape
            mask_gc[0:5, :] = cv2.GC_BGD  # Top edge
            mask_gc[-5:, :] = cv2.GC_BGD  # Bottom edge
            mask_gc[:, 0:5] = cv2.GC_BGD  # Left edge
            mask_gc[:, -5:] = cv2.GC_BGD  # Right edge
            
            # Step 5: Apply GrabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Run GrabCut with mask initialization
            cv2.grabCut(roi, mask_gc, None, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_MASK)
            
            # Create final mask
            final_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            
            # Step 6: Post-processing for smoother results
            # Fill holes and smooth edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Smooth the mask edges
            final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
            _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
            
            print(f"⚡ Applied Enhanced Canny+GrabCut hybrid ({np.sum(final_mask==255)/final_mask.size*100:.1f}% foreground)")
            return final_mask
            
        except Exception as e:
            print(f"⚠️ Enhanced Canny+GrabCut hybrid failed: {e}")
            # Fallback to enhanced Canny
            return self._canny_edge_segmentation(roi)
    
    def _metal_object_segmentation(self, roi: np.ndarray) -> np.ndarray:
        """Specialized segmentation for metal objects like screws"""
        try:
            # Step 1: Multi-channel analysis
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Step 2: Metal-specific preprocessing
            # Enhance contrast specifically for metallic objects
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            
            # Use L channel from LAB (good for brightness/metal detection)
            l_channel = lab[:,:,0]
            enhanced_l = clahe.apply(l_channel)
            
            # Step 3: Multiple thresholding approaches for metal detection
            # Otsu on enhanced grayscale
            _, otsu_mask = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Adaptive thresholding on L channel
            adaptive_mask = cv2.adaptiveThreshold(enhanced_l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 15, 5)
            
            # Triangle thresholding (good for bimodal histograms like metal vs background)
            _, triangle_mask = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            
            # Step 4: Combine masks using logical operations
            # Combine all three masks
            combined_mask = cv2.bitwise_or(cv2.bitwise_or(otsu_mask, adaptive_mask), triangle_mask)
            
            # Step 5: Morphological operations for metal objects
            # Use elliptical kernels (better for round screws)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            # Remove noise
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            
            # Fill holes in metal objects
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
            
            # Step 6: Contour filtering for screw-like shapes
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Filter contours based on area and shape
                total_area = roi.shape[0] * roi.shape[1]
                filtered_contours = []
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > total_area * 0.02:  # At least 2% of image area
                        # Check if shape is reasonable for a screw (not too elongated)
                        rect = cv2.boundingRect(contour)
                        aspect_ratio = max(rect[2], rect[3]) / min(rect[2], rect[3])
                        if aspect_ratio < 5:  # Not extremely elongated
                            filtered_contours.append(contour)
                
                # Create final mask from filtered contours
                if filtered_contours:
                    final_mask = np.zeros_like(gray)
                    cv2.drawContours(final_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
                else:
                    final_mask = combined_mask
            else:
                final_mask = combined_mask
            
            # Step 7: Final smoothing
            # Gaussian blur to smooth edges
            final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
            _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
            
            print(f"🔩 Applied Metal object segmentation ({np.sum(final_mask==255)/final_mask.size*100:.1f}% foreground)")
            return final_mask
            
        except Exception as e:
            print(f"⚠️ Metal object segmentation failed: {e}")
            return self._aggressive_threshold_segmentation(roi)
    
    def _evaluate_mask_quality(self, mask: np.ndarray, roi: np.ndarray) -> float:
        """
        Evaluate the quality of a segmentation mask - STRONGLY prefer screw preservation
        Conservative approach: Better to keep background than lose screw parts
        """
        if mask is None:
            return 0
        
        fg_pixels = np.sum(mask == 255)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        if total_pixels == 0:
            return 0
            
        fg_ratio = fg_pixels / total_pixels
        
        # CRITICAL CHECK: Validate if background removal is targeting outer areas, not inner
        h, w = mask.shape
        background_removal_score = self._validate_background_removal_direction(mask)
        if background_removal_score < 0.5:  # Failed validation
            print(f"    ❌ Background removal targeting inner parts instead of outer background")
            return 0.1  # Reject this segmentation
        
        # VERY conservative: Strongly penalize aggressive removal (prefer keeping screw intact)
        if fg_ratio < 0.3:  # Less than 30% remaining = too aggressive
            return 0.1  # Very low score
        elif fg_ratio < 0.4:  # 30-40% remaining = risky
            return 0.3
        elif fg_ratio >= 0.9:  # 90%+ remaining = too conservative (no background removal)
            return 0.4
        
        # Preferred range: 40-85% foreground (moderate background removal)
        if 0.4 <= fg_ratio <= 0.85:
            # Score increases with MORE foreground (more conservative)
            ratio_score = 0.5 + (fg_ratio - 0.4) * 1.1  # 0.5 to 1.0
        else:
            ratio_score = 0.4
            
        ratio_score = min(1.0, ratio_score)
        
        # Check mask connectivity (prefer single connected component = intact screw)
        num_components, labeled = cv2.connectedComponents(mask)
        if num_components <= 2:  # Background + 1 main object
            component_score = 1.0
        elif num_components <= 3:  # Background + 2 objects (still acceptable)
            component_score = 0.8
        else:
            component_score = 0.4  # Too fragmented - likely damaged screw
        
        # STRONGLY favor methods that preserve central areas (screws are usually centered)
        center_region = mask[h//4:3*h//4, w//4:3*w//4]
        center_fg_ratio = np.sum(center_region == 255) / center_region.size if center_region.size > 0 else 0
        
        # High penalty if center is removed (likely screw damage)
        if center_fg_ratio < 0.5:  # Less than 50% of center preserved
            center_score = 0.2  # Heavy penalty
        elif center_fg_ratio < 0.7:  # 50-70% of center preserved
            center_score = 0.6
        else:
            center_score = 1.0  # Good center preservation
        
        # Penalize masks that touch borders too much (likely incomplete segmentation)
        border_pixels = np.sum(mask[0, :]) + np.sum(mask[-1, :]) + np.sum(mask[:, 0]) + np.sum(mask[:, -1])
        border_ratio = border_pixels / (2 * (h + w - 2))
        
        if border_ratio > 0.8:  # Object touching 80%+ of borders
            border_score = 0.7  # Some penalty 
        else:
            border_score = 1.0
        
        total_score = ratio_score * component_score * center_score * border_score * background_removal_score
        
        # Debug info for quality assessment
        debug_info = f"fg={fg_ratio:.2f}, comp={num_components-1}, center={center_fg_ratio:.2f}, border={border_ratio:.2f}, bg_removal={background_removal_score:.2f}"
        if total_score < 0.5:
            print(f"    📉 Low quality: {debug_info}")
        
        return total_score
    
    def _validate_background_removal_direction(self, mask: np.ndarray) -> float:
        """
        Validate if background removal is targeting outer areas (good) vs inner areas (bad).
        Returns score 0-1 where higher means removal is correctly targeting outer background.
        """
        h, w = mask.shape
        
        # Get removed areas (background pixels that were removed)
        removed_mask = (mask == 0).astype(np.uint8)
        
        # Define outer regions (border areas where background should be)
        border_thickness = min(h, w) // 8  # Adjust based on image size
        outer_region = np.zeros_like(mask, dtype=bool)
        
        # Create outer region mask (borders)
        outer_region[:border_thickness, :] = True  # Top
        outer_region[-border_thickness:, :] = True  # Bottom
        outer_region[:, :border_thickness] = True  # Left
        outer_region[:, -border_thickness:] = True  # Right
        
        # Define inner region (center area where screw should be)
        inner_region = np.zeros_like(mask, dtype=bool)
        inner_start_h, inner_end_h = h//4, 3*h//4
        inner_start_w, inner_end_w = w//4, 3*w//4
        inner_region[inner_start_h:inner_end_h, inner_start_w:inner_end_w] = True
        
        # Count removed pixels in outer vs inner regions
        removed_in_outer = np.sum(removed_mask[outer_region])
        removed_in_inner = np.sum(removed_mask[inner_region])
        total_removed = np.sum(removed_mask)
        
        if total_removed == 0:
            return 1.0  # No removal = perfect (conservative)
        
        # Calculate ratios
        outer_removal_ratio = removed_in_outer / total_removed if total_removed > 0 else 0
        inner_removal_ratio = removed_in_inner / total_removed if total_removed > 0 else 0
        
        # Good removal should target outer areas more than inner areas
        if outer_removal_ratio > inner_removal_ratio:
            # More removal in outer regions = good
            ratio_score = outer_removal_ratio / (outer_removal_ratio + inner_removal_ratio + 0.01)
        else:
            # More removal in inner regions = bad (removing screw)
            ratio_score = 0.2  # Low score but not zero (might be salvageable)
        
        # Additional penalty if too much inner area is removed
        inner_region_total = np.sum(inner_region)
        if inner_region_total > 0:
            inner_damage_ratio = removed_in_inner / inner_region_total
            if inner_damage_ratio > 0.4:  # More than 40% of center removed
                ratio_score *= 0.3  # Heavy penalty
            elif inner_damage_ratio > 0.2:  # More than 20% of center removed
                ratio_score *= 0.6
        
        return min(1.0, ratio_score)

    def _apply_mask_with_transparency(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to create transparent background"""
        result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        result[:, :, :3] = image
        result[:, :, 3] = mask
        return result
    
    def polygon_to_axis_aligned_bbox(self, points: List[Tuple[float, float]], 
                                   img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Convert polygon points to axis-aligned bounding box
        
        This is the KEY improvement - convert OBB to proper rectangle
        """
        # Convert normalized coordinates to pixel coordinates
        pixel_points = []
        for x_norm, y_norm in points:
            x_px = int(x_norm * img_width)
            y_px = int(y_norm * img_height)
            pixel_points.append((x_px, y_px))
        
        # Find axis-aligned bounding box
        x_coords = [p[0] for p in pixel_points]
        y_coords = [p[1] for p in pixel_points]
        
        x1 = max(0, min(x_coords))
        y1 = max(0, min(y_coords))
        x2 = min(img_width - 1, max(x_coords))
        y2 = min(img_height - 1, max(y_coords))
        
        print(f"🔄 OBB->RECT: Polygon {len(points)} points -> Axis-aligned Rectangle ({x1}, {y1}, {x2}, {y2})")
        
        return (x1, y1, x2, y2)
    
    def extract_region_with_padding(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                                  padding: int = 0) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract rectangular region WITHOUT padding - exact screw size"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # NO PADDING - extract exact rectangle
        # Just ensure bounds are within image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extract exact region
        extracted = image[y1:y2, x1:x2]
        actual_bbox = (x1, y1, x2, y2)
        
        print(f"📏 Extracted region: {extracted.shape[1]}x{extracted.shape[0]} pixels (exact screw size, no padding)")
        
        return extracted, actual_bbox

    def create_proper_label(self, original_annotation: Dict, extraction_bbox: Tuple[int, int, int, int],
                          extracted_size: Tuple[int, int], original_size: Tuple[int, int], 
                          original_bbox: Tuple[int, int, int, int]) -> str:
        """
        Transform original polygon to fit extracted image exactly:
        - Min coordinate becomes 0, max coordinate becomes 1
        - Maintains original polygon shape proportions
        """
        class_id = original_annotation['class_id']
        extracted_w, extracted_h = extracted_size
        
        if original_annotation['type'] == 'bbox':
            # For bbox, create centered rectangle matching extraction
            center_x, center_y = 0.5, 0.5
            width = 0.8  # Use most of the extracted area
            height = 0.8
            
            label = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            print(f"📝 BBOX Label (FITTED): center=({center_x:.3f},{center_y:.3f}), size=({width:.3f},{height:.3f})")
            
        elif original_annotation['type'] == 'obb':
            # Get original normalized polygon points
            points = original_annotation['points']
            orig_w, orig_h = original_size
            
            # Convert normalized coordinates to pixel coordinates
            pixel_points = []
            for x_norm, y_norm in points:
                x_px = x_norm * orig_w
                y_px = y_norm * orig_h
                pixel_points.append((x_px, y_px))
            
            # Find the actual polygon bounds (this should match extraction_bbox)
            x_coords = [p[0] for p in pixel_points]
            y_coords = [p[1] for p in pixel_points]
            poly_x_min, poly_x_max = min(x_coords), max(x_coords)
            poly_y_min, poly_y_max = min(y_coords), max(y_coords)
            
            # Transform polygon coordinates to fit extracted image exactly
            # Min becomes 0, max becomes 1
            transformed_coords = []
            poly_width = poly_x_max - poly_x_min
            poly_height = poly_y_max - poly_y_min
            
            for x_px, y_px in pixel_points:
                # Normalize within polygon bounds: (coord - min) / (max - min)
                if poly_width > 0:
                    new_x = (x_px - poly_x_min) / poly_width
                else:
                    new_x = 0.5
                    
                if poly_height > 0:
                    new_y = (y_px - poly_y_min) / poly_height  
                else:
                    new_y = 0.5
                
                # Clamp to ensure valid coordinates
                new_x = max(0.0, min(1.0, new_x))
                new_y = max(0.0, min(1.0, new_y))
                
                transformed_coords.extend([new_x, new_y])
            
            # Format as polygon label
            coord_str = ' '.join([f"{coord:.6f}" for coord in transformed_coords])
            label = f"{class_id} {coord_str}"
            print(f"📝 POLYGON Label (NORMALIZED): Original bounds -> [0,1], {len(points)} points")
            
        return label
    
    def create_label_file_from_original(self, original_annotation: dict, extraction_bbox: Tuple[int, int, int, int], 
                                      extracted_image_size: Tuple[int, int], output_path: Path, 
                                      original_image_size: Tuple[int, int], original_bbox: Tuple[int, int, int, int]) -> bool:
        """
        Create YOLO format label file with proper coordinate transformation
        
        FIXED: Now properly transforms coordinates to fit within the extracted image
        
        Args:
            original_annotation: Original annotation from dataset
            extraction_bbox: The rectangular bbox used for extraction (x1, y1, x2, y2) in pixels WITH PADDING
            extracted_image_size: Size of extracted image (width, height) in pixels
            output_path: Path where to save the label file (.txt)
            original_image_size: Original full image size (width, height) in pixels
            original_bbox: Original screw bbox WITHOUT padding (x1, y1, x2, y2) in pixels
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create the properly fitted label content
            label_content = self.create_proper_label(
                original_annotation, extraction_bbox, extracted_image_size, 
                original_image_size, original_bbox
            )
            
            # Save to file
            with open(output_path, 'w') as f:
                f.write(label_content + '\n')
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating label file: {e}")
            return False
    
    def generate_filename(self, class_name: str, image_name: str, object_id: int) -> str:
        """
        Generate standardized filename for extracted screw
        
        Args:
            class_name: Class name of the screw
            image_name: Original image filename
            object_id: Object ID within the image
            
        Returns:
            Formatted filename
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        base_name = Path(image_name).stem
        return f"{class_name}_{base_name}_obj{object_id:02d}_{timestamp}.png"
    
    def extract_from_image(self, image_path: Path, method: str = "grabcut") -> int:
        """
        Extract all screws from a single image
        
        Args:
            image_path: Path to the image file
            method: Extraction method ('bbox', 'grabcut', 'obb')
            
        Returns:
            Number of screws extracted
        """
        # Find corresponding label file
        label_path = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
        
        if not label_path.exists():
            print(f"⚠️  No label file found for {image_path.name}")
            return 0
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return 0
        
        img_height, img_width = image.shape[:2]
        
        # Load annotations
        annotations = self.load_yolo_annotations(label_path)
        
        if not annotations:
            print(f"⚠️  No annotations found in {label_path.name}")
            return 0
        
        extracted_count = 0
        
        for i, annotation in enumerate(annotations):
            class_id = annotation['class_id']
            
            # Validate class ID
            if class_id >= len(self.class_names):
                print(f"⚠️  Invalid class ID {class_id} in {image_path.name}")
                continue
            
            class_name = self.class_names[class_id]
            
            try:
                # ===== IMPROVED 3-STEP EXTRACTION PROCESS =====
                # STEP 1: Convert annotation to proper bounding box
                pixel_coords = self.yolo_to_pixel_coords(annotation, img_height, img_width)
                
                if annotation['type'] == 'obb':
                    # Keep original polygon for labels, but create bounding box for extraction
                    original_bbox = self.polygon_to_axis_aligned_bbox(annotation['points'], img_width, img_height)
                    print(f"🔄 OBB EXTRACTION: Polygon {len(annotation['points'])} points -> Extraction Rectangle {original_bbox}")
                    # Keep original annotation for labels (don't convert to bbox)
                    original_annotation_for_labels = annotation  # Keep original polygon format
                else:  # bbox
                    # Already rectangular - convert normalized to pixel coordinates
                    original_bbox = pixel_coords
                    print(f"🔄 BBOX: Using rectangular {original_bbox}")
                    # Convert bbox annotation for labels
                    original_annotation_for_labels = {
                        'type': 'bbox',
                        'class_id': annotation['class_id'],
                        'center_x': annotation['center_x'],
                        'center_y': annotation['center_y'], 
                        'width': annotation['width'],
                        'height': annotation['height']
                    }
                
                # STEP 2: Extract rectangular region WITHOUT padding - exact screw size
                if method == "advanced":
                    # Extract exact screw rectangle without padding, then apply advanced background removal
                    extracted_region, extraction_bbox = self.extract_region_with_padding(image, original_bbox, padding=0)
                    
                    if extracted_region.size == 0:
                        print(f"⚠️ Empty region extracted, skipping")
                        continue
                        
                    # Apply advanced background removal to the padded region
                    processed_screw, bg_removal_pct = self.advanced_background_removal(extracted_region)
                    
                    if processed_screw is None:
                        print(f"⚠️ Advanced processing failed - skipping {class_name}")
                        continue  # Skip this screw entirely
                            
                elif method == "grabcut":
                    # Use GrabCut without padding - exact screw size
                    extracted_region, extraction_bbox = self.extract_region_with_padding(image, original_bbox, padding=0)
                    processed_screw = self.extract_with_grabcut(image, original_bbox)
                    if processed_screw.shape[:2] != extracted_region.shape[:2]:
                        # Resize to match extraction region if needed
                        processed_screw = cv2.resize(processed_screw, (extracted_region.shape[1], extracted_region.shape[0]))
                else:  # bbox method
                    # Simple rectangular extraction without padding - exact screw size
                    extracted_region, extraction_bbox = self.extract_region_with_padding(image, original_bbox, padding=0)
                    processed_screw = extracted_region
                    # Add alpha channel for consistency
                    if len(processed_screw.shape) == 3 and processed_screw.shape[2] == 3:
                        alpha = np.ones((processed_screw.shape[0], processed_screw.shape[1]), dtype=np.uint8) * 255
                        processed_screw = np.dstack([processed_screw, alpha])
                
                # STEP 3: Generate filename and save
                filename = self.generate_filename(class_name, image_path.name, i)
                image_output_file = self.output_path / class_name / "images" / filename
                label_output_file = self.output_path / class_name / "labels" / (Path(filename).stem + ".txt")
                
                # Save extracted screw image
                success = cv2.imwrite(str(image_output_file), processed_screw)
                
                # STEP 4: Create properly fitted label file
                extracted_size = (processed_screw.shape[1], processed_screw.shape[0])  # width, height
                original_img_size = (img_width, img_height)  # width, height
                
                # Create label using BOTH the padded extraction bbox AND the original screw bbox
                label_success = self.create_label_file_from_original(
                    original_annotation_for_labels, extraction_bbox, extracted_size, label_output_file, 
                    original_img_size, original_bbox
                )
                
                if success and label_success:
                    extracted_count += 1
                    self.extraction_stats[class_name] += 1
                    print(f"✅ Extracted {class_name} -> {filename} (with properly fitted label)")
                elif success:
                    extracted_count += 1
                    self.extraction_stats[class_name] += 1
                    print(f"⚠️  Extracted {class_name} -> {filename} (image only - label failed)")
                else:
                    print(f"❌ Failed to save {filename}")
                    self.failed_extractions.append(f"{image_path.name}:{class_name}:{i}")
                    
            except Exception as e:
                print(f"❌ Error extracting {class_name} from {image_path.name}: {e}")
                import traceback
                traceback.print_exc()
                self.failed_extractions.append(f"{image_path.name}:{class_name}:{i}:{str(e)}")
        
        return extracted_count
    
    def save_metadata(self, processed_images: List[str], method: str):
        """
        Save extraction metadata and statistics
        
        Args:
            processed_images: List of processed image filenames
            method: Extraction method used
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "output_path": str(self.output_path),
            "method": method,
            "processed_images": processed_images,
            "extraction_stats": self.extraction_stats,
            "failed_extractions": self.failed_extractions,
            "class_names": self.class_names,
            "total_extracted": sum(self.extraction_stats.values())
        }
        
        metadata_file = self.output_path / "_metadata" / f"extraction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📊 Metadata saved to {metadata_file}")
    
    def extract_sample(self, sample_size: int = 10, method: str = "grabcut", 
                      random_seed: Optional[int] = None) -> Dict:
        """
        Extract screws from a random sample of images
        
        Args:
            sample_size: Number of images to process
            method: Extraction method ('bbox', 'grabcut')
            random_seed: Random seed for reproducibility
            
        Returns:
            Extraction results summary
        """
        if random_seed is not None:
            random.seed(random_seed)
        else:
            # Use current timestamp for true randomization
            import time
            random.seed(int(time.time() * 1000) % (2**32))
        
        # Get all training images
        images_dir = self.dataset_path / "train" / "images"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Find all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(images_dir.glob(ext))
        
        if len(image_files) == 0:
            raise FileNotFoundError(f"No image files found in {images_dir}")
        
        # Always shuffle the list to ensure randomization
        random.shuffle(image_files)
        
        # Sample random images (or take first N if already shuffled)
        sample_images = image_files[:min(sample_size, len(image_files))]
        
        print(f"🎲 Randomized selection from {len(image_files)} available images")
        print(f"🔍 Processing {len(sample_images)} random images using {method} method...")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"📤 Output: {self.output_path}")
        print("=" * 70)
        
        total_extracted = 0
        processed_images = []
        
        for i, image_path in enumerate(sample_images, 1):
            print(f"📸 [{i}/{len(sample_images)}] Processing: {image_path.name}")
            
            extracted = self.extract_from_image(image_path, method)
            total_extracted += extracted
            processed_images.append(image_path.name)
            
            print(f"   ✨ {extracted} screws extracted from this image")
            print()
        
        # Save metadata
        self.save_metadata(processed_images, method)
        
        # Print summary
        self.print_summary()
        
        return {
            "total_extracted": total_extracted,
            "processed_images": len(sample_images),
            "extraction_stats": self.extraction_stats,
            "failed_extractions": len(self.failed_extractions)
        }
    
    def print_summary(self):
        """Print extraction summary statistics"""
        print("=" * 70)
        print("📊 EXTRACTION SUMMARY")
        print("=" * 70)
        
        total_extracted = sum(self.extraction_stats.values())
        print(f"🎯 Total screws extracted: {total_extracted}")
        print()
        
        print("📈 Breakdown by class:")
        for class_name, count in self.extraction_stats.items():
            if count > 0:
                print(f"   {class_name:20s}: {count:3d} screws")
        
        if self.failed_extractions:
            print(f"\n⚠️  Failed extractions: {len(self.failed_extractions)}")
            for failure in self.failed_extractions[:5]:  # Show first 5
                print(f"   - {failure}")
            if len(self.failed_extractions) > 5:
                print(f"   ... and {len(self.failed_extractions) - 5} more")
        
        print(f"\n📁 Extracted screws saved in: {self.output_path}")
        print("=" * 70)


def main():
    """Main execution function with command line interface"""
    parser = argparse.ArgumentParser(description="Extract screws from YOLO labeled images")
    parser.add_argument("--dataset", "-d", type=str, 
                       default="Final - training corrected.v1i.yolov11",
                       help="Path to YOLO dataset folder")
    parser.add_argument("--output", "-o", type=str, 
                       default="extracted_screws",
                       help="Output directory for extracted screws")
    parser.add_argument("--sample-size", "-n", type=int, 
                       default=10,
                       help="Number of random images to process")
    parser.add_argument("--method", "-m", type=str, 
                       choices=["bbox", "grabcut", "advanced"],
                       default="advanced",
                       help="Extraction method: bbox (simple crop), grabcut (GrabCut algorithm), advanced (multi-method segmentation with background removal and labels)")
    parser.add_argument("--seed", type=int,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = ScrewExtractor(args.dataset, args.output)
        
        # Extract sample
        results = extractor.extract_sample(
            sample_size=args.sample_size,
            method=args.method,
            random_seed=args.seed
        )
        
        print("🎉 Extraction completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    # Auto-run mode with fixed defaults
    import sys
    
    if len(sys.argv) == 1:
        print("🔧 Auto Screw Extractor")
        print("=" * 50)
        
        # Fixed defaults - no user interaction needed
        dataset_path = "Final - training corrected.v1i.yolov11"
        sample_size = 50
        method = 'advanced'  # Changed to 'grabcut'
        
        print(f"🚀 Starting extraction...")
        print(f"   📂 Dataset: {dataset_path}")
        print(f"   📊 Sample: {sample_size} images")
        print(f"   🔧 Method: {method}")
        print()
        
        extractor = ScrewExtractor(dataset_path)
        results = extractor.extract_sample(sample_size=sample_size, method=method)
        
    else:
        main()
