import av
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import torch
import os
import math
from pypylon import pylon
from collections import Counter
import matplotlib.pyplot as plt
import time
from datetime import datetime
from PIL import Image
# Set Streamlit layout to wide
st.set_page_config(layout="wide")

# Store camera state in session state
if 'camera' not in st.session_state:
    st.session_state.camera = None
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None
# ==============================SIDEBAR START========================================
st.sidebar.title("Settings")
# Check if CUDA is available and if enable the choice for 'cuda'
if torch.cuda.is_available():
    device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
else:
    device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)


# # Check if GPU is available
if torch.cuda.is_available():
    device_option = 'cuda'
else:
    device_option = 'cpu'
    st.sidebar.warning("No GPU available. Inference will run on CPU, which may cause lower performance!")

# Load weight files
weight_files = []

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create a relative path to the "checkpoints" folder
directory = os.path.join(script_dir, 'checkpoints')

for subdir, _, files in os.walk(directory):
    for file in files:
        if file.endswith(".pt"):  # Assuming weight files have .pt extension
            weight_files.append(file)

if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = None 

if weight_files:
    model_option = st.sidebar.radio("Select Model", weight_files, index=0)

    if model_option != st.session_state['selected_model']:
        st.session_state['selected_model'] = model_option 
        st.query_params.update(model=model_option) 
else:
    st.sidebar.write("No model weights found in the directory!")
    model_option = None

if st.session_state['selected_model']:
    model_option = os.path.join(script_dir, "checkpoints", st.session_state['selected_model'])
    model = YOLO(model_option)
    model.to(device_option)
    names = model.names
else:
    st.stop()

# Confidence slider
confidence = st.sidebar.slider('Confidence Threshold', min_value=0.1, max_value=1.0, value=0.8)

# Camera offset sliders
st.sidebar.subheader("Camera Settings")
offset_x = st.sidebar.slider('Camera OffsetX', min_value=524-100, max_value=524+100, value=524, step=2,
                            help="Adjust horizontal camera offset (default: 524, must be even)")
offset_y = st.sidebar.slider('Camera OffsetY', min_value=956-100, max_value=956+100, value=956, step=2,
                            help="Adjust vertical camera offset (default: 956, must be even)")

# Store offset values in session state to track changes
if 'current_offset_x' not in st.session_state:
    st.session_state.current_offset_x = 524
if 'current_offset_y' not in st.session_state:
    st.session_state.current_offset_y = 956

# st.text('Purlpe = Missing part')
# st.text('Green = Correct part')
# st.text('Red = Misplaced part')

# ==============================SIDEBAR END========================================
# Properly close the camera connection
def release_camera():
    try:
        if st.session_state.camera is not None and st.session_state.camera.IsOpen():
            st.session_state.camera.Close()
            st.info("Successfully released the Basler camera connection.")
    except Exception as e:
        st.warning(f"Failed to release camera: {e}")
    st.session_state.camera = None

# Initialize Basler camera with proper resource handling
def initialize_basler_camera():
    if st.session_state.camera is not None and st.session_state.camera.IsOpen():
        return st.session_state.camera
    release_camera()  # Ensure to release any existing camera connections

    try:
        st.session_state.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        st.session_state.camera.Open()
        st.session_state.camera.PixelFormat.Value = 'RGB8'
        st.session_state.camera.Width.Value = 3076 #3044#2472
        st.session_state.camera.Height.Value = 1852 #1904#1396
        # st.session_state.camera.OffsetX.value = 520 #800
        # st.session_state.camera.OffsetY.value = 580#184
        st.session_state.camera.OffsetX.SetValue(524) #Default offset X
        st.session_state.camera.OffsetY.SetValue(956) #Default offset Y
        st.session_state.camera.ExposureAuto.Value = "Once"
        st.session_state.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        st.success("Basler Camera initialized successfully.")
        return st.session_state.camera 
    except Exception as e:
        st.error(f"Failed to initialize Basler camera: {str(e)}")
        return None

def get_basler_image(camera):
    """Retrieve an image from the Basler camera and convert it to RGB format."""
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grab_result.GrabSucceeded():
        img = grab_result.Array
        img = cv2.rotate(img, cv2.ROTATE_180)
        grab_result.Release()
        img_rgb = img #cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    else:
        return None

def draw_label(image, label, region_start, region_end, img_height, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.5, color=(0, 0, 0), thickness=3):
    """Draw a label at the center of the specified region in the image."""
    region_center_x = (region_start + region_end) // 2
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x = region_center_x - text_size[0] // 2
    text_y = 50  # Adjust Y position based on image height to keep text near the top of the region
    cv2.putText(image, label, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

import cv2

def draw_label(image, label, region_start, region_end, img_height, 
               font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.5, 
               color=(0, 0, 0), thickness=3, background_color=(255, 255, 255), 
               padding=10):
    """Draw a label with a background at the center of the specified region in the image."""

    # Find the center of the region
    region_center_x = (region_start + region_end) // 2

    # Get the text size
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_w, text_h = text_size

    # Calculate the position of the text
    text_x = region_center_x - text_size[0] // 2
    text_y = 50  # Adjust Y position based on image height to keep text near the top of the region

    # Add padding to the rectangle
    rect_x1 = text_x - padding
    rect_y1 = text_y - text_h - padding
    rect_x2 = text_x + text_w + padding
    rect_y2 = text_y + padding

    # Ensure the rectangle does not exceed the image boundaries
    rect_x1 = max(0, rect_x1)
    rect_y1 = max(0, rect_y1)
    rect_x2 = min(image.shape[1], rect_x2)
    rect_y2 = min(image.shape[0], rect_y2)

    # Draw the background rectangle
    cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, -1)

    # Draw the text on top of the rectangle
    cv2.putText(image, label, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)


def draw_label_on_obb(image, label, obb, color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    """Draw a label at the center of the oriented bounding box (OBB) on the image."""
    x_center, y_center, width, height, angle = obb
    rect = ((float(x_center), float(y_center)), (float(width), float(height)), math.degrees(angle))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color, thickness)
    # Compute the label position (bottom-left corner of the bounding box)
    text_x, text_y = int(box[1][0]), int(box[1][1]) - 10
    cv2.putText(image, label, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

import cv2

def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=3,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(255, 255, 255),
              padding=10  # Padding to increase rectangle size
              ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    # Adjust the rectangle size by adding padding
    cv2.rectangle(img, 
                  (x - padding, y - padding),  # Top-left corner with padding
                  (x + text_w + padding, y + text_h + padding),  # Bottom-right corner with padding
                  text_color_bg, -1)

    # Adjust text position to be inside the padded rectangle
    cv2.putText(img, 
                text, 
                (x, y + text_h),  # Position adjusted by text height (no need for extra -1 here)
                font, font_scale, text_color, font_thickness)



    return text_size

def draw_text_on_image(image, text, position, color=(0, 0, 0), font_scale=1, thickness=2):
    """
    Draw text on the image at the specified position.

    Args:
        image (numpy.ndarray): The image on which to draw the text.
        text (str): The text to be drawn.
        position (tuple): The (x, y) position for the text on the image.
        color (tuple): The color of the text in (B, G, R) format. Default is black.
        font_scale (int): Font scale (size) of the text. Default is 1.
        thickness (int): Thickness of the text. Default is 2.
    """
    # Define font type
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the position for bottom-left corner
    position_final = (position[0], image.shape[0] - position[1])  # 10 pixels from the bottom-left corner

    # Draw the text on the image
    cv2.putText(image, text, position_final, font, font_scale, color, thickness, lineType=cv2.LINE_AA)


class YOLOVideoProcessor:
    def __init__(self):
        self.confidence = confidence
        self.device = device_option
        self.camera = initialize_basler_camera()

    def update_camera_offset(self):
        """Update camera offset if slider values have changed"""
        if (self.camera is not None and self.camera.IsOpen()):
            # Check if offset values have changed
            if (st.session_state.current_offset_x != offset_x or 
                st.session_state.current_offset_y != offset_y):
                try:
                    # Ensure values are even (required by Basler camera)
                    safe_offset_x = offset_x if offset_x % 2 == 0 else offset_x - 1
                    safe_offset_y = offset_y if offset_y % 2 == 0 else offset_y - 1
                    
                    # Stop grabbing temporarily
                    if self.camera.IsGrabbing():
                        self.camera.StopGrabbing()
                    
                    # Update offset values
                    self.camera.OffsetX.SetValue(safe_offset_x)
                    self.camera.OffsetY.SetValue(safe_offset_y)
                    
                    # Update session state
                    st.session_state.current_offset_x = offset_x
                    st.session_state.current_offset_y = offset_y
                    
                    # Restart grabbing
                    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    
                    if safe_offset_x != offset_x or safe_offset_y != offset_y:
                        st.sidebar.warning(f"Camera offset adjusted to even values: X={safe_offset_x}, Y={safe_offset_y}")
                    else:
                        st.sidebar.success(f"Camera offset updated: X={safe_offset_x}, Y={safe_offset_y}")
                except Exception as e:
                    st.sidebar.error(f"Failed to update camera offset: {e}")
                    # Try to restart grabbing if it was stopped
                    try:
                        if not self.camera.IsGrabbing():
                            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    except:
                        pass

    def recv(self):
        img = get_basler_image(self.camera)
        if img is None:
            return None, [], [], []

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = img.shape
        left_boundary = img_width // 3
        right_boundary = (img_width // 3) * 2

        results = model.predict(img, conf=self.confidence)
        inventory_objects, master_objects, assembly_objects = [], [], []
        inventory_obbs, assembly_obbs, master_obbs = {}, {}, {}

        for result in results:
            obbs = result.obb.xywhr
            confidences = result.obb.conf
            class_labels = result.obb.cls

            if obbs is not None and confidences is not None and class_labels is not None:
                for i, obb in enumerate(obbs):
                    confidence = confidences[i] if i < len(confidences) else 0.0
                    class_label = int(class_labels[i].item()) if i < len(class_labels) else -1
                    class_name = model.names[class_label] if class_label in range(len(model.names)) else "Unknown"

                    if len(obb) != 5:
                        continue

                    x_center, y_center, width, height, angle = obb

                    if x_center < left_boundary:
                        inventory_objects.append(class_name)
                        if class_name not in inventory_obbs:
                            inventory_obbs[class_name] = []
                        inventory_obbs[class_name].append(obb)
                    elif left_boundary <= x_center < right_boundary:
                        master_objects.append(class_name)
                        if class_name not in master_obbs:
                            master_obbs[class_name] = []
                        master_obbs[class_name].append(obb)
                    else:
                        assembly_objects.append(class_name)
                        if class_name not in assembly_obbs:
                            assembly_obbs[class_name] = []
                        assembly_obbs[class_name].append(obb)

        master_count, assembly_count = Counter(master_objects), Counter(assembly_objects)

        missing_items = {item: master_count[item] - assembly_count.get(item, 0) for item in master_count if master_count[item] > assembly_count.get(item, 0)}
        extra_items = {item: assembly_count[item] - master_count.get(item, 0) for item in assembly_count if assembly_count[item] > master_count.get(item, 0)}
        matching_items = {item for item in master_count if item in assembly_count}
  
        for item, obbs in assembly_obbs.items():
            num_master = master_count.get(item, 0)
            num_assembly = len(obbs)

            obbs_sorted = sorted(obbs, key=lambda obb: obb[0]) 

            correct_count = min(num_master, num_assembly)
            extra_count = num_assembly - num_master if num_assembly > num_master else 0

            for i in range(correct_count):
                draw_label_on_obb(img, item, obbs_sorted[i], color=(0, 255, 0))  

            for i in range(correct_count, num_assembly):
                draw_label_on_obb(img, item, obbs_sorted[i], color=(255, 0, 0)) 

          # Draw OBB for missing items in Inventory region and highlight if not found
        for item, count in missing_items.items():
            if item in inventory_obbs:
                # Draw OBB for missing items found in Inventory Table
                for obb in inventory_obbs[item]:
                    draw_label_on_obb(img, item, obb, color=(255, 0, 255))  # Blue box for missing items found in Inventory region
            else:
                # Indicate missing parts that are not found in Inventory
                draw_text_on_image(img, f"Part Not Found: {item}", (10, 30), color=(0, 0, 255), font_scale=1, thickness=2)
        if len(missing_items.items())==0 and len(extra_items.items())==0:
            draw_text(img, f"Kit Complete",pos=(1900, 700))
            #draw_text_on_image(img, f"Kit Complete", (1900, 700), color=(0, 255, 0), font_scale=2, thickness=4)
        for item, obbs in master_obbs.items():
            for obb in obbs:
                color = (0, 255, 0)  # Green for master objects
                draw_label_on_obb(img, item, obb, color)

        draw_label(img, "1. Inventar", 0, left_boundary, img_height)
        draw_label(img, "2. Muster-Bausatz", left_boundary, right_boundary, img_height)
        draw_label(img, "3. Montageplatz", right_boundary, img_width, img_height)

        # Convert image to video frame
        return av.VideoFrame.from_ndarray(img, format="rgb24"), inventory_objects, master_objects, assembly_objects

col1, col2 = st.columns([4,1])    
#st.title("Objekterkennung für Vollständigkeitsprüfung")
#col1.subheader("Objekterkennung für Vollständigkeitsprüfung")
with col1:
    st.header("Objekterkennung für Vollständigkeitsprüfung")

with col2:
    st.image("tu-berlin-logo-long-red.svg", width=200)

processor = YOLOVideoProcessor()

frame_placeholder = st.empty()
inventory_region, master_region, assembly_region = st.columns(3)

# Set placeholders in session state to persist between reruns
if 'inventory_placeholder' not in st.session_state:
    st.session_state['inventory_placeholder'] = inventory_region.empty()
if 'master_placeholder' not in st.session_state:
    st.session_state['master_placeholder'] = master_region.empty()
if 'assembly_placeholder' not in st.session_state:
    st.session_state['assembly_placeholder'] = assembly_region.empty()
if 'correct_parts_placeholder' not in st.session_state:
    st.session_state['correct_parts_placeholder'] = assembly_region.empty()
if 'missing_parts_placeholder' not in st.session_state:
    st.session_state['missing_parts_placeholder'] = assembly_region.empty()
if 'incorrect_parts_placeholder' not in st.session_state:
    st.session_state['incorrect_parts_placeholder'] = assembly_region.empty()

# new placeholder for standing-part message
if 'standing_placeholder' not in st.session_state:
    st.session_state['standing_placeholder'] = assembly_region.empty()

# Access placeholders using session state
inventory_placeholder = st.session_state['inventory_placeholder']
master_placeholder = st.session_state['master_placeholder']
assembly_placeholder = st.session_state['assembly_placeholder']
correct_parts_placeholder = st.session_state['correct_parts_placeholder']
missing_parts_placeholder = st.session_state['missing_parts_placeholder']
incorrect_parts_placeholder = st.session_state['incorrect_parts_placeholder']

# new local reference
standing_placeholder = st.session_state['standing_placeholder']

# Add this function before your main while loop

def save_screenshot(frame):
    if frame is None:
        return False, "No frame available to capture"
    
    try:
        # Create screenshots directory if it doesn't exist
        screenshot_dir = os.path.join(script_dir, 'screenshots')
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")
        
        # Save the image
        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return True, f"Screenshot saved as {os.path.basename(filename)}"
    except Exception as e:
        return False, f"Error saving screenshot: {str(e)}"
    
# Add a function to save raw camera image 
def save_raw_image():
    """Capture and save a raw image directly from the camera without any processing or overlays"""
    if processor.camera is None or not processor.camera.IsOpen():
        return False, "Camera is not initialized"
    
    try:
        # Get raw image from camera
        raw_img = get_basler_image(processor.camera)
        if raw_img is None:
            return False, "Failed to capture image from camera"
        
        # Create screenshots directory if it doesn't exist
        screenshot_dir = os.path.join(script_dir, 'raw_screenshots')
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(screenshot_dir, f"raw_image_{timestamp}.png")
        
        # Save the raw image using PIL to maintain RGB format
        raw_pil = Image.fromarray(raw_img)
        raw_pil.save(filename)
        return True, f"Raw image saved as {os.path.basename(filename)}"
    except Exception as e:
        return False, f"Error saving raw image: {str(e)}"

# Add screenshot button
screenshot_col, raw_col, feedback_col = st.columns([1, 1, 2])

with screenshot_col:
    if st.button("📸 Take Screenshot"):
        if st.session_state.last_frame is not None:
            success, message = save_screenshot(st.session_state.last_frame)
            if success:
                feedback_col.success(message)
            else:
                feedback_col.error(message)
        else:
            feedback_col.warning("No camera frame available")

with raw_col:
    if st.button("📷 Raw Camera Image"):
        success, message = save_raw_image()
        if success:
            feedback_col.success(message)
        else:
            feedback_col.error(message)

while True:
    if processor.camera is None:
        st.warning("Basler Camera is not initialized. Please check the connection and restart the program.")
        break

    # Update camera offset if sliders changed
    processor.update_camera_offset()

    frame, inventory_objects, master_objects, assembly_objects = processor.recv()
    if frame:
        # Store the latest frame in session state
        st.session_state.last_frame = frame.to_ndarray(format="rgb24")
        
        frame_placeholder.image(st.session_state.last_frame, channels="RGB")

        inventory_count = Counter(inventory_objects)
        master_count = Counter(master_objects)
        assembly_count = Counter(assembly_objects)

        inventory_display = "<br>".join([f"{cls}: {count}" for cls, count in inventory_count.items()])
       
        inventory_placeholder.markdown(
            f"""
            <span style='font-size:24px; font-weight:bold;'>Detected Parts</span><br>
            <span style='font-size:16px;'>{inventory_display}</span>
            """,
            unsafe_allow_html=True
        )


        master_display = "<br>".join([f"{cls}: {count}" for cls, count in master_count.items()])
        master_placeholder.markdown(
            f"""
            <span style='font-size:24px; font-weight:bold;'>Detected Parts</span><br>
            <span style='font-size:16px;'>{master_display}</span>
            """,
            unsafe_allow_html=True
        )
        

        missing_items = {item: master_count[item] - assembly_count.get(item, 0) for item in master_count if master_count[item] > assembly_count.get(item, 0)}
        extra_items = {item: assembly_count[item] - master_count.get(item, 0) for item in assembly_count if assembly_count[item] > master_count.get(item, 0)}
        matching_items = {item for item in master_count if item in assembly_count}
        correct_items = {item: min(master_count[item], assembly_count[item]) for item in matching_items}

    
        correct_content = ""
        if correct_items:
            for item, count in correct_items.items():
                correct_content += f'<span style="color:green; font-size:16px;">{item}: {count} correct</span><br>'
        correct_parts_placeholder.markdown(
            f"""
            <span style='font-size:24px; font-weight:bold;'>Correct Parts</span><br>
            <span style='font-size:16px;'>{correct_content}</span>
            """,
            unsafe_allow_html=True,
        )

        missing_content = ""
        if missing_items:
            for item, count in missing_items.items():
                missing_content += f'<span style="color:black; font-size:16px;">{item}: {count} missing</span><br>'
        missing_parts_placeholder.markdown(
            f"""
            <span style='font-size:24px; font-weight:bold;'>Missing Parts</span><br>
            <span style='font-size:16px;'>{missing_content}</span>
            """,
            unsafe_allow_html=True,
        )
        
        incorrect_content = ""
        if extra_items:
            for item, count in extra_items.items():
                incorrect_content += f'<span style="color:red; font-size:16px;">{item}: {count} extra</span><br>'
        incorrect_parts_placeholder.markdown(
            f"""
            <span style='font-size:24px; font-weight:bold;'>Incorrect Parts</span><br>
            <span style='font-size:16px;'>{incorrect_content}</span>
            """,
            unsafe_allow_html=True,
        )



    time.sleep(0.1)
