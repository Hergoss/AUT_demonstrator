import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
from pathlib import Path

class BackgroundPatternCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("Background Pattern Creator")
        self.root.geometry("1200x800")
        
        # Canvas dimensions
        self.target_width = 3076
        self.target_height = 1852
        
        # Variables
        self.original_image = None
        self.scaled_image = None
        self.preview_image = None
        self.comparison_image = None
        self.scale_factor = tk.DoubleVar(value=1.0)
        
        # Pattern offset controls
        self.offset_x = tk.IntVar(value=0)
        self.offset_y = tk.IntVar(value=0)
        
        # Canvas dimensions
        self.canvas_width = tk.IntVar(value=800)
        self.canvas_height = tk.IntVar(value=600)
        
        # Comparison overlay
        self.comparison_overlay = None
        self.comparison_x = 50  # Position in target coordinates
        self.comparison_y = 50
        self.comparison_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Mouse resize state variables
        self.resizing = False
        self.resize_edge = None
        self.start_x = 0
        self.start_y = 0
        self.start_width = 0
        self.start_height = 0
        self.resize_threshold = 8  # pixels from edge to show resize cursor
        
        # Default directory for background images
        self.default_dir = Path("echteHintergründe/Original Backgrounds")
        if not self.default_dir.exists():
            self.default_dir = Path("echteHintergründe")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding="5")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Button(file_frame, text="Select Image", command=self.select_image).grid(row=0, column=0, padx=(0, 10))
        self.file_label = ttk.Label(file_frame, text="No image selected", foreground="gray")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Controls & Comparison", padding="5")
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)
        
        # Scale slider
        ttk.Label(controls_frame, text="Scale Factor:").grid(row=0, column=0, padx=(0, 10), sticky=tk.W)
        scale_frame = ttk.Frame(controls_frame)
        scale_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))
        scale_frame.columnconfigure(0, weight=1)
        
        self.scale_slider = ttk.Scale(scale_frame, from_=0.1, to=5.0, variable=self.scale_factor, 
                                     orient=tk.HORIZONTAL, command=self.on_scale_change)
        self.scale_slider.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.scale_label = ttk.Label(scale_frame, text="1.0")
        self.scale_label.grid(row=0, column=1)
        
        # Pattern offset controls
        ttk.Label(controls_frame, text="Pattern Offset X:").grid(row=1, column=0, padx=(0, 10), sticky=tk.W)
        offset_x_frame = ttk.Frame(controls_frame)
        offset_x_frame.grid(row=1, column=1, sticky=(tk.W, tk.E))
        offset_x_frame.columnconfigure(0, weight=1)
        
        self.offset_x_slider = ttk.Scale(offset_x_frame, from_=0, to=500, variable=self.offset_x, 
                                        orient=tk.HORIZONTAL, command=self.on_offset_change)
        self.offset_x_slider.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.offset_x_label = ttk.Label(offset_x_frame, text="0")
        self.offset_x_label.grid(row=0, column=1)
        
        ttk.Label(controls_frame, text="Pattern Offset Y:").grid(row=2, column=0, padx=(0, 10), sticky=tk.W)
        offset_y_frame = ttk.Frame(controls_frame)
        offset_y_frame.grid(row=2, column=1, sticky=(tk.W, tk.E))
        offset_y_frame.columnconfigure(0, weight=1)
        
        self.offset_y_slider = ttk.Scale(offset_y_frame, from_=0, to=500, variable=self.offset_y, 
                                        orient=tk.HORIZONTAL, command=self.on_offset_change)
        self.offset_y_slider.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.offset_y_label = ttk.Label(offset_y_frame, text="0")
        self.offset_y_label.grid(row=0, column=1)
        
        # Reset and Save buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W,))
        
        ttk.Button(button_frame, text="Reset Scale", command=self.reset_scale).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Save Pattern", command=self.save_pattern).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Button(button_frame, text="Load Comparison Image", 
                  command=self.select_comparison_image).pack(side=tk.LEFT, padx=(0, 10))
        self.comparison_label = ttk.Label(button_frame, text="No comparison image", foreground="gray")
        self.comparison_label.pack(side=tk.LEFT)
        
        # Content frame for preview
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Preview frame with mouse resizing
        preview_frame = ttk.LabelFrame(content_frame, text="Preview (Hover edges to resize canvas, drag comparison overlay to move)", padding="5")
        preview_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # Canvas for preview (with scrollbars)
        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Create canvas with dynamic size and mouse events
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind mouse events for resizing and comparison overlay dragging
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Scrollbars
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Set initial canvas size
        self.canvas.configure(width=self.canvas_width.get(), height=self.canvas_height.get())
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready. Please select an image to begin.", 
                                     relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def select_image(self):
        """Open file dialog to select an image"""
        initial_dir = self.default_dir if self.default_dir.exists() else Path.cwd()
        
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Background Image",
            initialdir=str(initial_dir),
            filetypes=filetypes
        )
        
        if filename:
            try:
                # Load image
                self.original_image = Image.open(filename)
                self.file_label.config(text=os.path.basename(filename), foreground="black")
                self.status_label.config(text=f"Loaded image: {os.path.basename(filename)} "
                                               f"({self.original_image.width}x{self.original_image.height})")
                
                # Reset scale and update preview
                self.scale_factor.set(1.0)
                self.update_preview()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
                self.status_label.config(text="Error loading image")
    
    def select_comparison_image(self):
        """Open file dialog to select a comparison image"""
        initial_dir = Path("Screws_Png") if Path("Screws_Png").exists() else Path.cwd()
        
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Comparison Image",
            initialdir=str(initial_dir),
            filetypes=filetypes
        )
        
        if filename:
            try:
                # Load comparison image
                self.comparison_image = Image.open(filename)
                self.comparison_label.config(text=os.path.basename(filename), foreground="black")
                
                # Update preview to show overlay
                self.update_preview()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load comparison image:\n{str(e)}")
    
    def is_point_in_comparison_overlay(self, x, y):
        """Check if a point is within the comparison overlay bounds"""
        if not self.comparison_image or not hasattr(self, 'current_preview_scale'):
            return False
        
        # Calculate overlay bounds in preview coordinates
        overlay_x = int(self.comparison_x * self.current_preview_scale)
        overlay_y = int(self.comparison_y * self.current_preview_scale)
        
        # Scale comparison image dimensions according to preview scale (1:1 to target dimensions)
        overlay_width = int(self.comparison_image.width * self.current_preview_scale)
        overlay_height = int(self.comparison_image.height * self.current_preview_scale)
        
        return (overlay_x <= x <= overlay_x + overlay_width and
                overlay_y <= y <= overlay_y + overlay_height)
    
    def get_resize_cursor_and_edge(self, widget, x, y):
        """Determine if mouse is near edge and return cursor type and edge"""
        w_width = widget.winfo_width()
        w_height = widget.winfo_height()
        threshold = self.resize_threshold
        
        # Check edges
        near_right = abs(x - w_width) <= threshold
        near_left = x <= threshold
        near_bottom = abs(y - w_height) <= threshold
        near_top = y <= threshold
        
        if near_right and near_bottom:
            return "bottom_right_corner", "se"
        elif near_left and near_bottom:
            return "bottom_left_corner", "sw"
        elif near_right and near_top:
            return "top_right_corner", "ne"
        elif near_left and near_top:
            return "top_left_corner", "nw"
        elif near_right:
            return "right_side", "e"
        elif near_left:
            return "left_side", "w"
        elif near_bottom:
            return "bottom_side", "s"
        elif near_top:
            return "top_side", "n"
        else:
            return "arrow", None
    
    def on_canvas_motion(self, event):
        """Handle mouse motion over canvas for resize cursor and comparison overlay highlight"""
        if not self.resizing and not self.comparison_dragging:
            # Check if mouse is over comparison overlay
            if self.is_point_in_comparison_overlay(event.x, event.y):
                self.canvas.configure(cursor="hand2")
            else:
                # Check for canvas resize cursor
                cursor, edge = self.get_resize_cursor_and_edge(self.canvas, event.x, event.y)
                self.canvas.configure(cursor=cursor)
    
    def on_canvas_click(self, event):
        """Handle mouse click on canvas for resizing or comparison dragging"""
        # Check if clicking on comparison overlay
        if self.is_point_in_comparison_overlay(event.x, event.y):
            self.comparison_dragging = True
            
            # Calculate drag offset in target coordinates
            if hasattr(self, 'current_preview_scale'):
                overlay_x = int(self.comparison_x * self.current_preview_scale)
                overlay_y = int(self.comparison_y * self.current_preview_scale)
                
                self.drag_start_x = (event.x - overlay_x) / self.current_preview_scale
                self.drag_start_y = (event.y - overlay_y) / self.current_preview_scale
        else:
            # Check for canvas resizing
            cursor, edge = self.get_resize_cursor_and_edge(self.canvas, event.x, event.y)
            if edge:
                self.resizing = True
                self.resize_edge = edge
                self.start_x = event.x_root
                self.start_y = event.y_root
                self.start_width = self.canvas_width.get()
                self.start_height = self.canvas_height.get()
    
    def on_canvas_drag(self, event):
        """Handle mouse drag for canvas resizing or comparison overlay movement"""
        if self.comparison_dragging and hasattr(self, 'current_preview_scale'):
            # Calculate new position in target coordinates
            new_x_target = (event.x / self.current_preview_scale) - self.drag_start_x
            new_y_target = (event.y / self.current_preview_scale) - self.drag_start_y
            
            # Clamp to boundaries (in target coordinates)
            # Use actual comparison image dimensions for boundary checking
            max_x = self.target_width - self.comparison_image.width
            max_y = self.target_height - self.comparison_image.height
            
            self.comparison_x = max(0, min(max_x, new_x_target))
            self.comparison_y = max(0, min(max_y, new_y_target))
            
            # Update preview to show new overlay position
            self.update_preview()
            
        elif self.resizing and self.resize_edge:
            # Resize canvas
            dx = event.x_root - self.start_x
            dy = event.y_root - self.start_y
            
            new_width = self.start_width
            new_height = self.start_height
            
            # Calculate new dimensions based on resize edge
            if "e" in self.resize_edge:  # Right edge
                new_width = max(400, min(1200, self.start_width + dx))
            elif "w" in self.resize_edge:  # Left edge
                new_width = max(400, min(1200, self.start_width - dx))
            
            if "s" in self.resize_edge:  # Bottom edge
                new_height = max(300, min(800, self.start_height + dy))
            elif "n" in self.resize_edge:  # Top edge
                new_height = max(300, min(800, self.start_height - dy))
            
            # Update canvas dimensions
            self.canvas_width.set(new_width)
            self.canvas_height.set(new_height)
            self.canvas.configure(width=new_width, height=new_height)
            
            # Update preview
            if self.original_image:
                self.update_preview()
    
    def on_canvas_release(self, event):
        """Handle mouse release"""
        self.resizing = False
        self.comparison_dragging = False
        self.resize_edge = None
        self.canvas.configure(cursor="arrow")
    
    def on_scale_change(self, value=None):
        """Handle scale slider change"""
        scale_value = self.scale_factor.get()
        self.scale_label.config(text=f"{scale_value:.2f}")
        
        if self.original_image:
            self.update_preview()
    
    def on_offset_change(self, value=None):
        """Handle offset slider change"""
        offset_x_value = int(self.offset_x.get())
        offset_y_value = int(self.offset_y.get())
        self.offset_x_label.config(text=str(offset_x_value))
        self.offset_y_label.config(text=str(offset_y_value))
        
        if self.original_image:
            self.update_preview()
    
    def reset_scale(self):
        """Reset scale factor to 1.0"""
        self.scale_factor.set(1.0)
        if self.original_image:
            self.update_preview()
    
    def update_preview(self):
        """Update the preview canvas with the scaled and tiled image and comparison overlay"""
        if not self.original_image:
            return
        
        try:
            # Scale the original image
            scale = self.scale_factor.get()
            new_width = int(self.original_image.width * scale)
            new_height = int(self.original_image.height * scale)
            
            if new_width <= 0 or new_height <= 0:
                return
            
            self.scaled_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create tiled pattern
            pattern = self.create_tiled_pattern(self.scaled_image, self.target_width, self.target_height, 
                                              self.offset_x.get(), self.offset_y.get())
            
            # Create preview (scaled down to fit in canvas)
            canvas_width = int(self.canvas_width.get())
            canvas_height = int(self.canvas_height.get())
            
            preview_scale = min(canvas_width / self.target_width, canvas_height / self.target_height)
            preview_width = int(self.target_width * preview_scale)
            preview_height = int(self.target_height * preview_scale)
            
            # Store preview scale for mouse event handling
            self.current_preview_scale = preview_scale
            self.current_preview_width = preview_width
            self.current_preview_height = preview_height
            
            self.preview_image = pattern.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and display
            photo = ImageTk.PhotoImage(self.preview_image)
            
            # Clear canvas and add new image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Keep a reference
            
            # Add comparison overlay if comparison image is loaded
            if self.comparison_image:
                self.add_comparison_overlay(preview_scale, preview_width, preview_height)
            
            # Update canvas scroll region
            self.canvas.configure(scrollregion=(0, 0, preview_width, preview_height))
            
            # Update status
            tiles_x = int(np.ceil(self.target_width / new_width))
            tiles_y = int(np.ceil(self.target_height / new_height))
            self.status_label.config(text=f"Preview updated. Scaled size: {new_width}x{new_height}, "
                                          f"Tiles: {tiles_x}x{tiles_y}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update preview:\n{str(e)}")
            self.status_label.config(text="Error updating preview")
    
    def add_comparison_overlay(self, preview_scale, preview_width, preview_height):
        """Add the comparison image as an overlay on the canvas"""
        if not self.comparison_image:
            return
        
        try:
            # Calculate overlay position in preview coordinates
            overlay_x = int(self.comparison_x * preview_scale)
            overlay_y = int(self.comparison_y * preview_scale)
            
            # Scale comparison image dimensions according to preview scale (1:1 to target dimensions)
            overlay_width = int(self.comparison_image.width * preview_scale)
            overlay_height = int(self.comparison_image.height * preview_scale)
            
            # Make sure overlay stays within preview bounds
            overlay_x = min(overlay_x, preview_width - overlay_width)
            overlay_y = min(overlay_y, preview_height - overlay_height)
            overlay_x = max(0, overlay_x)
            overlay_y = max(0, overlay_y)
            
            # Scale comparison image to match preview scale (1:1 to target dimensions)
            comp_resized = self.comparison_image.resize((overlay_width, overlay_height), Image.Resampling.LANCZOS)
            comp_photo = ImageTk.PhotoImage(comp_resized)
            
            # Create overlay on canvas with border
            self.canvas.create_rectangle(overlay_x-2, overlay_y-2, 
                                       overlay_x + overlay_width + 2, overlay_y + overlay_height + 2,
                                       outline="red", width=2, tags="comparison_overlay")
            
            self.canvas.create_image(overlay_x, overlay_y, anchor=tk.NW, image=comp_photo, tags="comparison_overlay")
            
            # Keep reference to avoid garbage collection
            self.comparison_overlay = comp_photo
            
            # Add text label
            self.canvas.create_text(overlay_x + overlay_width//2, overlay_y + overlay_height + 15,
                                  text="Comparison (1:1 to target)", fill="red", font=("Arial", 10, "bold"),
                                  tags="comparison_overlay")
            
        except Exception as e:
            print(f"Error adding comparison overlay: {str(e)}")
    
    def create_tiled_pattern(self, image, target_width, target_height, offset_x=0, offset_y=0):
        """Create a tiled pattern from the given image with optional offset"""
        img_width, img_height = image.size
        
        # Calculate how many tiles we need (with extra for offset)
        tiles_x = int(np.ceil((target_width + offset_x) / img_width)) + 1
        tiles_y = int(np.ceil((target_height + offset_y) / img_height)) + 1
        
        # Create a larger canvas that can accommodate all tiles
        canvas_width = tiles_x * img_width
        canvas_height = tiles_y * img_height
        
        # Create new image with proper mode
        if image.mode == 'RGBA':
            pattern = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))
        else:
            pattern = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        
        # Tile the image starting from offset position
        for y in range(tiles_y):
            for x in range(tiles_x):
                x_pos = x * img_width - offset_x
                y_pos = y * img_height - offset_y
                pattern.paste(image, (x_pos, y_pos))
        
        # Crop to exact target size
        pattern = pattern.crop((0, 0, target_width, target_height))
        
        return pattern
        
        return pattern
    
    def save_pattern(self):
        """Save the current pattern to a file"""
        if not self.original_image:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        
        try:
            # Create full resolution pattern
            scale = self.scale_factor.get()
            new_width = int(self.original_image.width * scale)
            new_height = int(self.original_image.height * scale)
            
            if new_width <= 0 or new_height <= 0:
                messagebox.showerror("Error", "Invalid scale factor!")
                return
            
            scaled_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            pattern = self.create_tiled_pattern(scaled_image, self.target_width, self.target_height, 
                                              self.offset_x.get(), self.offset_y.get())
            
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                title="Save Pattern",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ]
            )
            
            if filename:
                # Convert to RGB if saving as JPEG
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    if pattern.mode == 'RGBA':
                        # Create white background
                        rgb_pattern = Image.new('RGB', pattern.size, (255, 255, 255))
                        rgb_pattern.paste(pattern, mask=pattern.split()[3] if pattern.mode == 'RGBA' else None)
                        pattern = rgb_pattern
                
                pattern.save(filename)
                messagebox.showinfo("Success", f"Pattern saved successfully!\nFile: {os.path.basename(filename)}")
                self.status_label.config(text=f"Pattern saved: {os.path.basename(filename)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save pattern:\n{str(e)}")
            self.status_label.config(text="Error saving pattern")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = BackgroundPatternCreator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
