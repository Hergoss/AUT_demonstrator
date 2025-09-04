#%% original to 640x640 yolo image comparison
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to the image
image_path = r"C:\Users\henri\Desktop\AUT Projekt\Demo\05_Kit_Verificator\metrics_newCamera_labeled\auflosung1_png.rf.a31b4f37ef8b1debf9860e3f7a1d1e62.jpg"

# Load the image (BGR format)
image = cv2.imread(image_path)

# Get original dimensions
orig_h, orig_w = image.shape[:2]

# Determine the scale factor and new size
scale = 640.0 / max(orig_w, orig_h)
new_w, new_h = int(orig_w * scale), int(orig_h * scale)

# Resize image with aspect ratio kept
resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# Compute padding
pad_w = 640 - new_w
pad_h = 640 - new_h

top = pad_h // 2
bottom = pad_h - top
left = pad_w // 2
right = pad_w - left

# Pad image (letterboxing)
letterboxed_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# Convert images to RGB for matplotlib
original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
letterboxed_rgb = cv2.cvtColor(letterboxed_image, cv2.COLOR_BGR2RGB)

# Display both images side by side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title(f"Original Image ({orig_w}x{orig_h})")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(letterboxed_rgb)
plt.title("Letterboxed Image (640x640)")
plt.axis("off")

plt.tight_layout()

# Save the figure
output_path = r"C:\Users\henri\Desktop\letterboxing_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Show the figure
plt.show()