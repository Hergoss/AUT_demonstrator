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
import background_creation as bg_creation
import matplotlib.pyplot as plt

#%% old manuel code

    # #%% artifical image generation
    # mask_dir = 'Masks'  # Directory containing the mask images
    # # setput image generation
    # label_dir = 'test_generated_labels' #Create a directory for labels
    # os.makedirs(label_dir, exist_ok=True)
    # label_path = os.path.join(label_dir, 'generated_image_labels.txt')
    # number_generated_objects = 30 # Number of masks to placed in the new image
    # image_name = 'generated_image.png' # Name of the generated image

    # # 1) Read in mask images and their names
    # mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    # masks = []
    # mask_names = []
    # for path in mask_paths:
    #     mask_img = io.imread(path)
    #     masks.append(mask_img)
    #     mask_names.append(os.path.basename(path))

    # # 2) Create an empty RGB image of size 3076 x 1852
    # canvas_h, canvas_w = 1852, 3076
    # canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # # 3) Choose 10 random masks, randomly position and rotate them in the image
    # num_masks = number_generated_objects
    # chosen_indices = random.choices(range(len(masks)), k=num_masks)
    # placed_info = []

    # for idx in chosen_indices:
    #     mask = masks[idx]
    #     mask_name = mask_names[idx]
    #     mask_h, mask_w = mask.shape

    #     # Random rotation
    #     angle = random.uniform(0, 360)
    #     rotated_mask = rotate(mask, angle, resize=True, preserve_range=True).astype(np.uint8)
    #     rot_h, rot_w = rotated_mask.shape

    #     # Random position
    #     max_y = canvas_h - rot_h
    #     max_x = canvas_w - rot_w
    #     if max_y < 0 or max_x < 0:
    #         continue  # skip if mask doesn't fit

    #     rand_y = random.randint(0, max_y)
    #     rand_x = random.randint(0, max_x)

    #     # Place mask on canvas using the actual intensity values
    #     for c in range(3):
    #         canvas[rand_y:rand_y+rot_h, rand_x:rand_x+rot_w, c] = np.maximum(
    #             canvas[rand_y:rand_y+rot_h, rand_x:rand_x+rot_w, c],
    #             rotated_mask
    #         )

    #     placed_info.append({
    #         'mask': rotated_mask,
    #         'mask_name': mask_name,
    #         'rand_x': rand_x,
    #         'rand_y': rand_y
    #     })

    # # 4) Create a single label file in YOLOv8 OBB 8 points format using mask image name as label

    # with open(label_path, 'w') as f:
    #     for info in placed_info:
    #         mask = info['mask']
    #         mask_name = info['mask_name']
    #         rand_x = info['rand_x']
    #         rand_y = info['rand_y']

    #         contours = find_contours(mask, level=0.5)
    #         if not contours:
    #             continue
    #         contour = max(contours, key=lambda x: x.shape[0])  # largest contour
    #         poly = Polygon(contour[:, ::-1])  # (row, col) -> (x, y)
    #         if not poly.is_valid:
    #             poly = poly.buffer(0)
    #         min_rect = poly.minimum_rotated_rectangle
    #         x, y = min_rect.exterior.coords.xy
    #         obb_points = np.array(list(zip(x, y)))[:-1]  # Remove duplicate last point

    #         # Shift OBB points to canvas position and normalize
    #         points_n = []
    #         for pt in obb_points:
    #             px = (pt[0] + rand_x) / canvas_w
    #             py = (pt[1] + rand_y) / canvas_h
    #             points_n.extend([px, py])

    #         line = f'{mask_name} ' + ' '.join([str(x) for x in points_n])
    #         f.write(line + '\n')

    # # 5) Save and plot this image
    # canvas_path = os.path.join(label_dir, image_name)
    # io.imsave(canvas_path, canvas)

    # plt.figure(figsize=(10, 6))
    # plt.imshow(canvas)
    # plt.title('Masks Planted on Canvas')
    # plt.axis('off')
    # plt.show()

    # # 6) Use the label to create the OBB bounding box around the masks and show the label names
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(canvas)
    # with open(label_path, 'r') as f:
    #     for line in f:
    #         parts = line.strip().split()
    #         mask_name = parts[0]
    #         points_n = [float(x) for x in parts[1:]]
    #         points = np.array(points_n).reshape(4, 2)
    #         points[:, 0] *= canvas_w
    #         points[:, 1] *= canvas_h
    #         poly = np.vstack([points, points[0]])
    #         ax.plot(poly[:, 0], poly[:, 1], color='red', linewidth=2)
    #         centroid = np.mean(points, axis=0)
    #         ax.text(centroid[0], centroid[1], mask_name, color='yellow', fontsize=12, ha='center', va='center')
    # plt.title('OBB Bounding Boxes with Label Names')
    # plt.axis('off')
    # plt.show()
    # # Save the figure with OBB bounding boxes and label names
    # obb_image_path = os.path.join(label_dir, 'canvas_with_obb_labels.png')
    # fig.savefig(obb_image_path, bbox_inches='tight', pad_inches=0)


#%%
def create_screws_CanvasAndLabel(mask_dir, label_path, number_generated_objects=30):
    """Create images with screws
    Args:
        mask_dir (str): Directory containing the mask images
        label_path (str): Path to save the label file
        number_generated_objects (int): Number of screws/nuts/washers to be placed in the new image
    returns:
        np.ndarray: The generated screw canvas.
    """
  # 1) Read in mask images and their names
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    masks = []
    mask_names = []
    for path in mask_paths:
        mask_img = io.imread(path)
        masks.append(mask_img)
        mask_names.append(os.path.basename(path))

    # 2) Create an empty RGB image of size 3076 x 1852
    canvas_h, canvas_w = 1852, 3076
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 3) Choose 10 random masks, randomly position and rotate them in the image
    num_masks = number_generated_objects
    chosen_indices = random.choices(range(len(masks)), k=num_masks)
    placed_info = []

    for idx in chosen_indices:
        mask = masks[idx]
        mask_name = mask_names[idx]
        mask_h, mask_w = mask.shape

        # Random rotation
        angle = random.uniform(0, 360)
        rotated_mask = rotate(mask, angle, resize=True, preserve_range=True).astype(np.uint8)
        rot_h, rot_w = rotated_mask.shape

        # Random position
        max_y = canvas_h - rot_h
        max_x = canvas_w - rot_w
        if max_y < 0 or max_x < 0:
            continue  # skip if mask doesn't fit

        rand_y = random.randint(0, max_y)
        rand_x = random.randint(0, max_x)

        # Place mask on canvas using the actual intensity values
        for c in range(3):
            canvas[rand_y:rand_y+rot_h, rand_x:rand_x+rot_w, c] = np.maximum(
                canvas[rand_y:rand_y+rot_h, rand_x:rand_x+rot_w, c],
                rotated_mask
            )

        placed_info.append({
            'mask': rotated_mask,
            'mask_name': mask_name,
            'rand_x': rand_x,
            'rand_y': rand_y
        })

    # 4) Create a single label file in YOLOv8 OBB 8 points format using mask image name as label

    with open(label_path, 'w') as f:
        for info in placed_info:
            mask = info['mask']
            mask_name = info['mask_name']
            rand_x = info['rand_x']
            rand_y = info['rand_y']

            contours = find_contours(mask, level=0.5)
            if not contours:
                continue
            contour = max(contours, key=lambda x: x.shape[0])  # largest contour
            poly = Polygon(contour[:, ::-1])  # (row, col) -> (x, y)
            if not poly.is_valid:
                poly = poly.buffer(0)
            min_rect = poly.minimum_rotated_rectangle
            x, y = min_rect.exterior.coords.xy
            obb_points = np.array(list(zip(x, y)))[:-1]  # Remove duplicate last point

            # Shift OBB points to canvas position and normalize
            points_n = []
            for pt in obb_points:
                px = (pt[0] + rand_x) / canvas_w
                py = (pt[1] + rand_y) / canvas_h
                points_n.extend([px, py])
            # Extract class name from mask_name (e.g., "mask_0.png" -> "mask")
            class_name = mask_name.split('_')[0]
            line = f'{class_name } ' + ' '.join([str(x) for x in points_n])
            f.write(line + '\n')
    return canvas

def create_distrubance_Canvas(mask_dir='disturbance_masks', number_generated_objects=30):
    """Create images with screws
    Args:
        mask_dir (str): Directory containing the mask images
        number_generated_objects (int): Number of screws/nuts/washers to be placed in the new image
    returns:
        np.ndarray: The generated screw canvas.
    """
  # 1) Read in mask images and their names
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    masks = []
    mask_names = []
    for path in mask_paths:
        mask_img = io.imread(path)
        masks.append(mask_img)
        mask_names.append(os.path.basename(path))

    # 2) Create an empty RGB image of size 3076 x 1852
    canvas_h, canvas_w = 1852, 3076
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 3) Choose 10 random masks, randomly position and rotate them in the image
    num_masks = number_generated_objects
    chosen_indices = random.choices(range(len(masks)), k=num_masks)
    placed_info = []

    for idx in chosen_indices:
        mask = masks[idx]
        mask_name = mask_names[idx]
        mask_h, mask_w = mask.shape

        # Random rotation
        angle = random.uniform(0, 360)
        rotated_mask = rotate(mask, angle, resize=True, preserve_range=True).astype(np.uint8)
        rot_h, rot_w = rotated_mask.shape

        # Random position
        max_y = canvas_h - rot_h
        max_x = canvas_w - rot_w
        if max_y < 0 or max_x < 0:
            continue  # skip if mask doesn't fit

        rand_y = random.randint(0, max_y)
        rand_x = random.randint(0, max_x)

        # Place mask on canvas using the actual intensity values
        for c in range(3):
            canvas[rand_y:rand_y+rot_h, rand_x:rand_x+rot_w, c] = np.maximum(
                canvas[rand_y:rand_y+rot_h, rand_x:rand_x+rot_w, c],
                rotated_mask
            )

        placed_info.append({
            'mask': rotated_mask,
            'mask_name': mask_name,
            'rand_x': rand_x,
            'rand_y': rand_y
        })
    return canvas


# %%
def create_screwImage(mask_dir='Masks', ouput_dir='generated_ImageLabel', image_name = 'generated_image', number_generated_objects=40):
    """Create images with screws
    Args:
        mask_dir (str): Directory containing the mask images
        label_dir (str): Directory to save label and image file
        output_dir (str): Directory to save the generated images
        image_name (str): Name of the generated image and label file
        number_generated_objects (int): Number of screws/nuts/washers to be placed in the new image
    Returns:
        np.ndarray: The generated screw image.
    """

    # setput image generation
    os.makedirs(ouput_dir, exist_ok=True)
    label_path = os.path.join(ouput_dir, image_name+'.txt') #  Label file path
    generatied_imagePath = os.path.join(ouput_dir, image_name+'.png')

    # 1) Create the screw canvas and label
    canvas = create_screws_CanvasAndLabel(mask_dir, label_path, number_generated_objects)
    # 2) Create a background image with random shapes
    if random.random() < 0.5:
        background_image = bg_creation.create_simpleBackground_image(num_shapes=30)
    else:
        background_image = bg_creation.create_ObjectBackground_image(background_path='Backgrounds')

    # 2.5) Add other screws/nuts/washers to the background image
    disturbance_canvas = create_distrubance_Canvas(mask_dir='disturbance_masks', number_generated_objects=50)
    background_image = np.where(disturbance_canvas > 0, disturbance_canvas, background_image)

    # 3) integrate the background image with the canvas
    # If a pixel in canvas is not 0, use the canvas pixel intensity, otherwise use the background pixel intensity
    blended_image = np.where(canvas > 0, canvas, background_image)

    #  Apply a Gaussian filter to the blended image for smoothing
    blur_intensity = np.random.uniform(0.5, 2)  # Random blur intensity 
    blended_image = filters.gaussian(blended_image, sigma=blur_intensity, channel_axis=-1)
    blended_image = (blended_image * 255).astype(np.uint8) # Convert back to uint8 after smoothing

    #  Add random noise to the blended image
    noise = np.random.normal(0, 5, blended_image.shape).astype(np.uint8) # Mean 0, Std Dev 5
    blended_image = np.clip(blended_image + noise, 0, 255).astype(np.uint8)


    # 3) Save and plot this image
    io.imsave(generatied_imagePath, blended_image)

    return blended_image

def display_obb_with_labels(image_path, label_path):
    """
    Reads an image and its corresponding YOLOv8 OBB label file,
    then displays the image with OBB bounding boxes and class names.

    Args:
        image_path (str): Path to the generated image.
        label_path (str): Path to the label file in YOLOv8 OBB 8 points format.
    """
    try:
        image = io.imread(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    canvas_h, canvas_w, _ = image.shape

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_name = parts[0]
                points_n = [float(x) for x in parts[1:]]

                # Ensure there are enough points for an OBB (8 points for 4 corners)
                if len(points_n) == 8:
                    points = np.array(points_n).reshape(4, 2)
                    points[:, 0] *= canvas_w
                    points[:, 1] *= canvas_h
                    poly = np.vstack([points, points[0]])  # Close the polygon

                    ax.plot(poly[:, 0], poly[:, 1], color='red', linewidth=2)
                    centroid = np.mean(points, axis=0)
                    ax.text(centroid[0], centroid[1], class_name, color='yellow', fontsize=12, ha='center', va='center')
                else:
                    print(f"Warning: Skipping malformed line in label file: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Label file not found at {label_path}")
        return
    except Exception as e:
        print(f"An error occurred while processing the label file: {e}")
        return

    plt.title('OBB Bounding Boxes with Class Names')
    plt.axis('off')
    plt.show()
    return

# %%
if __name__ == "__main__":
    # test_image = create_screwImage()
    # plt.imshow(test_image)
    # plt.title('Generated Screw Image')
    # plt.axis('off')
    # plt.show()

    # Display the OBB bounding boxes with labels
    create_screwImage(mask_dir='Masks', ouput_dir='generated_ImageLabel', image_name = 'generated_image', number_generated_objects=40)
    display_obb_with_labels(image_path='generated_ImageLabel/generated_image.png', label_path='generated_ImageLabel/generated_image.txt')
  

# %%
