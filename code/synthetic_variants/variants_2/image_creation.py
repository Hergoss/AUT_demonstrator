#%%
import os
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours
from shapely.geometry import Polygon
import random
import glob
from skimage.transform import rotate
import time
import background_creation as bg_creation
import shutil

#%%
def create_screws_CanvasAndLabel(mask_dir, label_path, number_generated_objects=30):
    """Create images with screws and YOLOv8 OBB labels.

    Args:
        mask_dir (str): Directory containing the mask images
        label_path (str): Path to save the label file
        number_generated_objects (int): Number of screws/nuts/washers to be placed in the new image
    Returns:
        np.ndarray: The generated screw canvas.
    """
    # 1) Get mask image paths and names
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    mask_names = [os.path.basename(path) for path in mask_paths]

    # 2) Create an empty RGB image of size 3076 x 1852
    canvas_h, canvas_w = 1852, 3076
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 3 & 4) Open label file once and write labels as we go
    with open(label_path, 'w') as f:
        num_masks = number_generated_objects
        chosen_indices = random.choices(range(len(mask_paths)), k=num_masks)

        for idx in chosen_indices:
            mask_path = mask_paths[idx]
            mask_name = mask_names[idx]
            mask = io.imread(mask_path)

            # Ensure mask is RGB
            if mask.ndim == 2:
                mask = np.stack([mask]*3, axis=-1)
            elif mask.shape[2] == 4:
                mask = mask[..., :3]

            # Random rotation
            angle = random.uniform(0, 360)
            rotated_mask = rotate(mask, angle, resize=True, preserve_range=True).astype(np.uint8)
            rot_h, rot_w, _ = rotated_mask.shape

            # Random position
            max_y = canvas_h - rot_h
            max_x = canvas_w - rot_w
            if max_y < 0 or max_x < 0:
                continue  # skip if mask doesn't fit

            rand_y = random.randint(0, max_y)
            rand_x = random.randint(0, max_x)

            # Place mask on canvas
            for c in range(3):
                canvas[rand_y:rand_y+rot_h, rand_x:rand_x+rot_w, c] = np.maximum(
                    canvas[rand_y:rand_y+rot_h, rand_x:rand_x+rot_w, c],
                    rotated_mask[..., c]
                )

            # ---- Label creation happens here ----
            mask_gray = color.rgb2gray(rotated_mask)
            mask_bin = mask_gray > 0.001
            mask_bin = np.pad(mask_bin, pad_width=3, mode='constant', constant_values=0)
            contours = find_contours(mask_bin, level=0.5)
            if not contours:
                continue

            contour = max(contours, key=lambda x: x.shape[0])
            poly = Polygon(contour[:, ::-1])  # (row, col) -> (x, y)
            if not poly.is_valid:
                poly = poly.buffer(0)
            min_rect = poly.minimum_rotated_rectangle
            x, y = min_rect.exterior.coords.xy
            obb_points = np.array(list(zip(x, y)))[:-1]  # drop duplicate last point

            # Shift to canvas position & normalize
            points_n = []
            for pt in obb_points:
                px = (pt[0] + rand_x) / canvas_w
                py = (pt[1] + rand_y) / canvas_h
                points_n.extend([px, py])

            class_name = mask_name.split('_')[0]
            line = f'{class_name} ' + ' '.join([str(x) for x in points_n])
            f.write(line + '\n')

    return canvas


def create_distrubance_Canvas(mask_dir='disturbance_masks', number_generated_objects=30):
    """Create images with screws.

    Args:
        mask_dir (str): Directory containing the mask images
        number_generated_objects (int): Number of screws/nuts/washers (that are not a class) to be placed in the new image
    Returns:
        np.ndarray: The generated screw canvas.
    """
    # 1) Get mask image paths and names
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    mask_names = [os.path.basename(path) for path in mask_paths]

    # 2) Create an empty RGB image of size 3076 x 1852
    canvas_h, canvas_w = 1852, 3076
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 3) Choose random masks, randomly position and rotate them in the image
    num_masks = number_generated_objects
    chosen_indices = random.choices(range(len(mask_paths)), k=num_masks)
    placed_info = []

    for idx in chosen_indices:
        mask_path = mask_paths[idx]
        mask_name = mask_names[idx]
        mask_img = io.imread(mask_path)
        # Ensure mask is RGB
        if mask_img.ndim == 2:
            mask_img = np.stack([mask_img]*3, axis=-1)
        elif mask_img.shape[2] == 4:
            mask_img = mask_img[..., :3]
        mask_h, mask_w, _ = mask_img.shape

        # Random rotation
        angle = random.uniform(0, 360)
        rotated_mask = rotate(mask_img, angle, resize=True, preserve_range=True).astype(np.uint8)
        rot_h, rot_w, _ = rotated_mask.shape

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
                rotated_mask[..., c]
            )

    return canvas


# %%
def create_screwImage(mask_dir='Masks', output_dir='generated_ImageLabel', image_name = 'generated_image', number_generated_objects=40, number_generated_otherScrews=45, number_generated_shapes=10, simpleBackground=False, ImageSave=False, Png=True):
    """Create images with screws.
    
    Args:
        mask_dir (str): Directory containing the mask images
        output_dir (str): Directory to save the generated images
        image_name (str): Name of the generated image and label file
        number_generated_objects (int): Number of screws/nuts/washers to be placed in the new image
        number_generated_otherScrews (int): Number of other screws/nuts/washers to be placed in the new image
        number_generated_shapes (int): Number of random shapes to be placed in the new image
        simpleBackground (bool): Whether to add a simple white background (True) or with random non technical objectes (False)
        ImageSave (bool): Whether to save the generated image
        Png (bool): Whether to save the generated image as PNG (True) or JPG (False)
    Returns:
        np.ndarray: The generated screw image.
    """
    start_time = time.time()

    # setput image generation
    os.makedirs(output_dir, exist_ok=True)
    label_path = os.path.join(output_dir, image_name+'.txt') #  Label file path
    if Png:
        generatied_imagePath = os.path.join(output_dir, image_name+'.png')
    else:
        # If not PNG, save as JPG
        generatied_imagePath = os.path.join(output_dir, image_name+'.jpg')   

    #step1_start = time.time()
    # 1) Create the screw canvas and label
    screw_canvas = create_screws_CanvasAndLabel(mask_dir, label_path, number_generated_objects)
    #step1_end = time.time()
    #print(f"Step 1 (create_screws_CanvasAndLabel) took: {step1_end - step1_start:.2f} seconds")

    #step2_start = time.time()
    # 2) Create a background image with random shapes
    if simpleBackground:
            background_image = bg_creation.create_simpleBackground_image(num_shapes=number_generated_shapes)
    else:
        background_image = bg_creation.create_ObjectBackground_image(background_path=r'Backgrounds')
    #step2_end = time.time()
    #print(f"Step 2 (create background canvas) took: {step2_end - step2_start:.2f} seconds")


    # step2_5_start = time.time()
    # 2.5) Add other screws/nuts/washers to the background image
    disturbance_canvas = create_distrubance_Canvas(mask_dir='disturbance_masks', number_generated_objects=number_generated_otherScrews)
    # Both disturbance_canvas and background_image are RGB, so blend per channel
    mask = disturbance_canvas.sum(axis=-1, keepdims=True) > 0
    background_image = np.where(mask, disturbance_canvas, background_image)
    #step2_5_end = time.time()
    #print(f"Step 2.5 (create and add other screws to backgraound canvas) took: {step2_5_end - step2_5_start:.2f} seconds")

    # Step 3a: Integrate the background image with the canvas
    #step3a_start = time.time()
    # Both blending_mask and canvas are RGB images
    blending_mask = np.any(screw_canvas > 0, axis=-1, keepdims=True)
    blended_image = np.where(blending_mask, screw_canvas, background_image)
    #step3a_end = time.time()
    #print(f"Step 3a (add screw canvas to background canvas) took: {step3a_end - step3a_start:.2f} seconds")

    # # Step 3b: Apply Gaussian smoothing
    # step3b_start = time.time()
    # blur_intensity = np.random.uniform(0.5, 2)  # Random blur intensity
    # ksize = int(6 * blur_intensity) | 1         # Ensure kernel size is odd
    # blended_image = cv2.GaussianBlur(blended_image, (ksize, ksize), blur_intensity)
    # step3b_end = time.time()
    # print(f"Step 3b (smoothing) took: {step3b_end - step3b_start:.2f} seconds")

    # # Step 3c: Add random noise
    # step3c_start = time.time()
    # blended_image = blended_image.astype(np.float32)  # Convert to float for noise addition
    # noise = np.random.normal(0, 5, blended_image.shape)
    # blended_image += noise
    # np.clip(blended_image, 0, 255, out=blended_image)
    # blended_image = blended_image.astype(np.uint8)
    # step3c_end = time.time()
    # print(f"Step 3c (adding noise) took: {step3c_end - step3c_start:.2f} seconds")



    # 3) Save the image
    if ImageSave:
        io.imsave(generatied_imagePath, blended_image)
        print(f"Image saved to {generatied_imagePath}")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
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

#%% label name correction 
# Define the class mapping
class_mapping = {
    "M4-12mm": 0,
    "M4-16mm": 1,
    "M4-20mm": 2,
    "M4-8mm": 3,
    "M4-Nut": 4,
    "M4-Washer": 5,
    "M6-12mm": 6,
    "M6-16mm": 7,
    "M6-20mm": 8,
    "M6-8mm": 9,
    "M6-Nut": 10,
    "M6-Washer": 11,
    "Standing-Nut": 12,
    "Standing-Screw": 13
}

def convert_labels_to_indices(label_folder="./generated_ImageLabel"):
    # Walk through the folder
    for filename in os.listdir(label_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(label_folder, filename)
            new_lines = []
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_name = parts[0]
                        if class_name in class_mapping:
                            class_index = class_mapping[class_name]
                            new_line = f"{class_index} " + " ".join(parts[1:]) + "\n"
                            new_lines.append(new_line)
                        else:
                            print(f"[WARNING] Unknown class name '{class_name}' in file {filename}")
            
            # Write the converted labels back to the file
            with open(file_path, 'w') as f:
                f.writelines(new_lines)


def organize_generated_files(base_folder="./generated_ImageLabel"):
    # Define paths for subfolders
    images_folder = os.path.join(base_folder, "images")
    labels_folder = os.path.join(base_folder, "labels")

    # Create subfolders if they don't exist
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # Move files
    for filename in os.listdir(base_folder):
        file_path = os.path.join(base_folder, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                shutil.move(file_path, os.path.join(images_folder, filename))
            elif ext == '.txt':
                shutil.move(file_path, os.path.join(labels_folder, filename))

def find_labels_without_images(base_folder="./generated_ImageLabel"):
    labels_folder = os.path.join(base_folder, "labels")
    images_folder = os.path.join(base_folder, "images")

    # Get set of image basenames without extensions
    image_files = os.listdir(images_folder)
    image_basenames = {os.path.splitext(f)[0] for f in image_files if os.path.isfile(os.path.join(images_folder, f))}

    # Check each label file if corresponding image exists
    missing_images = []
    for label_file in os.listdir(labels_folder):
        if label_file.endswith(".txt"):
            label_basename = os.path.splitext(label_file)[0]
            if label_basename not in image_basenames:
                missing_images.append(label_file)

    if missing_images:
        print("Labels without corresponding images:")
        for lbl in missing_images:
            print(f" - {lbl}")
    else:
        print("All label files have corresponding images.")

#%% functions for heatmap plot in analyis

def generate_screw_position_image(mask_dir='positionMasks', output_dir='positionImages', x_pos=10, y_pos=10, angle=0, Png = True):
    """Create images with screws and YOLOv8 OBB labels.

    Args:
        mask_dir (str): Directory containing the mask images
        output_dir (str): Directory to save the generated images and labels
        image_name (str): Name of the generated image and label file
        x_pos (int): X position to place the object
        y_pos (int): Y position to place the object
        angle (int): Angle to rotate the object (0-360°)
    Returns:

    """
    # 1) Get mask image paths and names
    os.makedirs(output_dir, exist_ok=True)

    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    mask_names = [os.path.basename(path) for path in mask_paths]

    # 3 & 4) Open label file once and write labels as we go
    for idx in range(len(mask_paths)):
        name_ding = mask_names[idx].split('_')[0]
        image_name = f'{name_ding}_at_xpos{x_pos}_ypos{y_pos}_angle{angle}'
        label_path = os.path.join(output_dir, image_name+'.txt') #  Label file path
        if Png:
            generated_imagePath = os.path.join(output_dir, image_name+'.png')
        else:
            # If not PNG, save as JPG
            generated_imagePath = os.path.join(output_dir, image_name+'.jpg')   

        # 2) Create an empty RGB image of size 3076 x 1852
        canvas_h, canvas_w = 1852, 3076
        canvas = np.full((canvas_h, canvas_w, 3), 0, dtype=np.uint8)

        with open(label_path, 'w') as f:
            mask_path = mask_paths[idx]
            mask_name = mask_names[idx]
            mask = io.imread(mask_path)

            # Ensure mask is RGB
            if mask.ndim == 2:
                mask = np.stack([mask]*3, axis=-1)
            elif mask.shape[2] == 4:
                mask = mask[..., :3]

            # Random rotation
            rotated_mask = rotate(mask, angle, resize=True, preserve_range=True).astype(np.uint8)
            rot_h, rot_w, _ = rotated_mask.shape

            # Random position
            max_y = canvas_h - rot_h
            max_x = canvas_w - rot_w
            if max_y < 0 or max_x < 0:
                continue  # skip if mask doesn't fit

            rand_y = min(y_pos, max_y)
            rand_x = min(x_pos, max_x)

            # Place mask on canvas
            for c in range(3):
                canvas[rand_y:rand_y+rot_h, rand_x:rand_x+rot_w, c] = np.maximum(
                    canvas[rand_y:rand_y+rot_h, rand_x:rand_x+rot_w, c],
                    rotated_mask[..., c]
                )

            # ---- Label creation happens here ----
            mask_gray = color.rgb2gray(rotated_mask)
            mask_bin = mask_gray > 0.001
            mask_bin = np.pad(mask_bin, pad_width=3, mode='constant', constant_values=0)
            contours = find_contours(mask_bin, level=0.5)
            if not contours:
                continue

            contour = max(contours, key=lambda x: x.shape[0])
            poly = Polygon(contour[:, ::-1])  # (row, col) -> (x, y)
            if not poly.is_valid:
                poly = poly.buffer(0)
            min_rect = poly.minimum_rotated_rectangle
            x, y = min_rect.exterior.coords.xy
            obb_points = np.array(list(zip(x, y)))[:-1]  # drop duplicate last point

            # Shift to canvas position & normalize
            points_n = []
            for pt in obb_points:
                px = (pt[0] + rand_x) / canvas_w
                py = (pt[1] + rand_y) / canvas_h
                points_n.extend([px, py])

            class_name = mask_name.split('_')[0]
            line = f'{class_name} ' + ' '.join([str(x) for x in points_n])
            f.write(line + '\n')

        # Make all pixels that are not part of the screw white
        white_background = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        canvas = np.where(canvas == 0, white_background, canvas)
        io.imsave(generated_imagePath, canvas)
        print(f"Image saved to {generated_imagePath}")

    #checks and other stuff
    # convert_labels_to_indices(label_folder=f"./{output_dir}")
    # organize_generated_files(base_folder=f"./{output_dir}")
    # find_labels_without_images(base_folder=f"./{output_dir}")

    return 

# %%
if __name__ == "__main__":
    # test_image = create_screwImage()
    # plt.imshow(test_image)
    # plt.title('Generated Screw Image')
    # plt.axis('off')
    # plt.show()

    # Display the OBB bounding boxes with labels
    create_screwImage(mask_dir='Masks', output_dir='generated_ImageLabel', image_name = 'generated_image', number_generated_objects=40, ImageSave=True)
    display_obb_with_labels(image_path='generated_ImageLabel/generated_image.png', label_path='generated_ImageLabel/generated_image.txt')

    # Generate a screw at a specific position (used in analysis in the heatmap plot)
    #generate_screw_position_image(mask_dir='positionMasks', output_dir='positionImages', x_pos=10, y_pos=10, angle=0, Png = True)
    for x in range(0, 3076, 100):
        for y in range(0, 1852, 100):
            angle = 0
            generate_screw_position_image(mask_dir='positionMasks', output_dir='positionImages', x_pos=x, y_pos=y, angle=angle, Png = True)
    convert_labels_to_indices(label_folder=f"./positionImages")
    organize_generated_files(base_folder=f"./positionImages")
    find_labels_without_images(base_folder=f"./positionImages")
# %%
