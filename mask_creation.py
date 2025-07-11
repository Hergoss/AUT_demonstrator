#%%
import os
from skimage import io, color
from skimage import feature
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure
import numpy as np
from skimage.measure import find_contours
from skimage.draw import polygon
from skimage.measure import regionprops
from shapely.geometry import Polygon
import math
import random
import glob
from skimage.transform import rotate

# Path to the images folder
images_folder = 'images'

# List to store grayscale images
gray_images = []

# Read and convert images
for filename in os.listdir(images_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(images_folder, filename)
        img = io.imread(img_path)
        if img.ndim == 3:  # Color image
            gray_img = color.rgb2gray(img)
        else:  # Already grayscale
            gray_img = img
        gray_images.append(gray_img)
#%%
# # Apply a threshold of 0.6 to the first image
# threshold = 0.6
# binary_image = gray_images[0] > threshold

# # Optionally, display the result
# plt.imshow(binary_image, cmap='gray')
# plt.title('Thresholded Image (0.6)')
# plt.axis('off')
# plt.show()

# # Hyperparameters for Canny edge detector
# canny_sigma = 1.0  # Standard deviation of the Gaussian filter
# canny_low_threshold = None  # Lower bound for hysteresis thresholding (None uses default)
# canny_high_threshold = None  # Upper bound for hysteresis thresholding (None uses default)

# # Apply Canny edge detector to the first grayscale image
# edges = feature.canny(
#     gray_images[0],
#     sigma=canny_sigma,
#     low_threshold=canny_low_threshold,
#     high_threshold=canny_high_threshold
# )

# # Display the result
# plt.imshow(edges, cmap='gray')
# plt.title('Canny Edge Detection')
# plt.axis('off')
# plt.show()
#%% mask creation
################ make masks and set up directory 
################
# Create a directory for masks
to_be_masked = gray_images[3]
output_dir = 'masks_vorbereitet'
os.makedirs(output_dir, exist_ok=True)
minimum_size_mask = 40  # Minimum size for masks ...x... pixels


###############
############### otsu + contours
###############

# Step 1: Smooth image
blurred = filters.gaussian(to_be_masked, sigma=1.0)

# Step 2: Threshold
thresh_val = filters.threshold_otsu(blurred)
binary = blurred > thresh_val

# Step 3: Morphological operations
cleaned = morphology.remove_small_objects(binary, min_size=700)
cleaned = morphology.closing(cleaned, morphology.disk(5))


plt.imshow(cleaned, cmap='gray')

# Find contours at a constant value of 0.5
contours = find_contours(cleaned, level=0.5)

# Plot each contour on the original image
fig, ax = plt.subplots()
ax.imshow(to_be_masked, cmap='gray')

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=1)

plt.title('Contours of Screws')
plt.axis('off')
plt.show()

#%%
#####################
################## Mask Creation
####################
# Prepare masks for all contours larger than 40x40 pixels
all_masks = []
valid_contours = []
for contour in contours:
    mask = np.zeros_like(to_be_masked, dtype=np.uint8)
    rr, cc = polygon(contour[:, 0], contour[:, 1], mask.shape)
    mask[rr, cc] = 1
    # Find bounding box of the mask
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        continue
    minr, minc = np.min(rows), np.min(cols)
    maxr, maxc = np.max(rows), np.max(cols)
    height = maxr - minr + 1
    width = maxc - minc + 1
    if height >= minimum_size_mask and width >= minimum_size_mask:    #### check for minimum size!!!!!
        all_masks.append(mask)
        valid_contours.append(contour)

# Remove inner contours from outer masks
final_masks = []
for i, outer_mask in enumerate(all_masks):
    mask = outer_mask.copy()
    for j, inner_mask in enumerate(all_masks):
        if i != j:
            # If inner_mask is completely inside outer_mask, subtract it
            if np.all((inner_mask == 1) <= (outer_mask == 1)) and np.any(inner_mask == 1):
                if np.any((outer_mask - inner_mask) == 1):
                    mask = mask - inner_mask
    mask = np.clip(mask, 0, 1)
    final_masks.append(mask)

# Save cropped masked images
for i, mask in enumerate(final_masks):
    rr, cc = np.where(mask)
    if rr.size == 0 or cc.size == 0:
        continue  # skip empty masks
    minr, minc = np.min(rr), np.min(cc)
    maxr, maxc = np.max(rr), np.max(cc)
    height = maxr - minr + 1
    width = maxc - minc + 1
    if height < minimum_size_mask or width < minimum_size_mask:
        continue  # skip small masks
    cropped_img = to_be_masked[minr:maxr+1, minc:maxc+1]
    cropped_mask = mask[minr:maxr+1, minc:maxc+1]
    masked_img = cropped_img * cropped_mask
    out_path = os.path.join(output_dir, f'mask_{i}.png')
    io.imsave(out_path, (masked_img * 255).astype(np.uint8))

# Show one example if available
example_path = os.path.join(output_dir, 'mask_0.png')
if os.path.exists(example_path):
    example = io.imread(example_path)
    plt.figure()
    plt.imshow(example, cmap='gray')
    plt.title('Example Masked Contour')
    plt.axis('off')
    plt.show()

# Plot each valid contour on the original image with mask filenames
fig, ax = plt.subplots()
ax.imshow(to_be_masked, cmap='gray')

for i, contour in enumerate(valid_contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
    centroid = np.mean(contour, axis=0)
    mask_filename = f'mask_{i}.png'
    ax.text(centroid[1], centroid[0], mask_filename, color='yellow', fontsize=8, ha='center', va='center')

plt.title('Contours with Mask Filenames')
plt.axis('off')
plt.show()
# %%
