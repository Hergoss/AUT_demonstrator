#%%
import os
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon, disk, ellipse
from skimage.transform import resize
import cv2


if __name__ == "__main__":
    # Image size
    height, width = 1852, 3076

    # Create almost white (light gray) background
    background_int = np.random.randint(200, 255)
    background_color = [background_int, background_int, background_int]  # gray background
    image = np.ones((height, width, 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)

    # Number of random shapes (set this variable as needed)
    num_shapes = 20

    for _ in range(num_shapes):
        shape_type = np.random.choice(['rectangle', 'circle', 'ellipse', 'polygon'])
        color = [np.random.randint(180, 220) for _ in range(3)]  # random gray

        if shape_type == 'rectangle':
            x1, y1 = np.random.randint(0, width-200), np.random.randint(0, height-200)
            x2, y2 = x1 + np.random.randint(50, 400), y1 + np.random.randint(50, 400)
            x2 = np.clip(x2, 0, width - 1)
            y2 = np.clip(y2, 0, height - 1)
            rr, cc = polygon([y1, y1, y2, y2], [x1, x2, x2, x1])
            image[rr, cc] = color

        elif shape_type == 'circle':
            radius = np.random.randint(30, 200)
            cy, cx = np.random.randint(radius, height-radius), np.random.randint(radius, width-radius)
            y, x = np.ogrid[-radius:radius, -radius:radius]
            mask = x**2 + y**2 <= radius**2
            for dy in range(-radius, radius):
                for dx in range(-radius, radius):
                    if dx**2 + dy**2 <= radius**2:
                        yy, xx = cy + dy, cx + dx
                        if 0 <= yy < height and 0 <= xx < width:
                            image[yy, xx] = color

        elif shape_type == 'ellipse':

            cy, cx = np.random.randint(0, height - 1), np.random.randint(0, width - 1)
            ry, rx = np.random.randint(30, 200), np.random.randint(30, 200)
            orientation = np.random.uniform(0, 2 * np.pi)
            rr, cc = ellipse(cy, cx, ry, rx, shape=image.shape, rotation=orientation)
            image[rr, cc] = color

        elif shape_type == 'polygon':
            num_vertices = np.random.randint(3, 8)
            center_x = np.random.randint(0, width-1)
            center_y = np.random.randint(0, height-1)
            angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False)
            radii = np.random.randint(40, 200, size=num_vertices)
            xs = center_x + (radii * np.cos(angles)).astype(int)
            ys = center_y + (radii * np.sin(angles)).astype(int)
            xs = np.clip(xs, 0, width-1)
            ys = np.clip(ys, 0, height-1)
            rr, cc = polygon(ys, xs)
            image[rr, cc] = color

    # # Smooth the edges by applying a Gaussian filter
    # image = filters.gaussian(image, sigma=2, channel_axis=2)
    # image = (image * 255).astype(np.uint8) # Convert back to uint8 after smoothing

    # Show the image
    plt.figure(figsize=(15, 8))
    plt.imshow(image)



# %%


def create_simpleBackground_image(num_shapes=20):
    """
    Create a background image with random shapes using OpenCV.

    Args:
        num_shapes (int): Number of random shapes to draw.

    Returns:
        np.ndarray: The generated background image.
    """
    # Image size
    height, width = 1852, 3076

    # Create almost white (light gray) background
    background_int = np.random.randint(230, 256)
    background_color = [background_int] * 3
    image = np.full((height, width, 3), background_color, dtype=np.uint8)

    for _ in range(num_shapes):
        shape_type = np.random.choice(['rectangle', 'circle', 'ellipse', 'polygon'])
        color = [np.random.randint(180, 220) for _ in range(3)]  # light gray color

        if shape_type == 'rectangle':
            max_width = 400
            max_height = 400
            x1 = np.random.randint(0, width - max_width)
            y1 = np.random.randint(0, height - max_height)
            rect_width = np.random.randint(50, max_width)
            rect_height = np.random.randint(50, max_height)
            x2 = x1 + rect_width
            y2 = y1 + rect_height
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=-1)

        elif shape_type == 'circle':
            radius = np.random.randint(30, 200)
            cx = np.random.randint(radius, width - radius)
            cy = np.random.randint(radius, height - radius)
            cv2.circle(image, (cx, cy), radius, color, thickness=-1)

        elif shape_type == 'ellipse':
            cx = np.random.randint(0, width)
            cy = np.random.randint(0, height)
            ax1 = np.random.randint(30, 200)
            ax2 = np.random.randint(30, 200)
            angle = np.random.randint(0, 360)
            cv2.ellipse(image, (cx, cy), (ax1, ax2), angle, 0, 360, color, thickness=-1)

        elif shape_type == 'polygon':
            num_vertices = np.random.randint(3, 8)
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
            radii = np.random.randint(40, 200, size=num_vertices)
            pts = np.array([
                (
                    int(center_x + r * np.cos(a)),
                    int(center_y + r * np.sin(a))
                )
                for r, a in zip(radii, angles)
            ], np.int32)
            pts = np.clip(pts, [0, 0], [width - 1, height - 1])
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(image, [pts], color)

    return image

# %%
def create_ObjectBackground_image(background_path='Backgrounds'):
    """    Create a background image with random shapes.
    This function generates a background image with random shapes such as rectangles, circles, ellipses, and polygons.
    Args:
        background_path (str): Path to the directory containing background images.
    
    Returns:
        np.ndarray: The generated background image.
    """

    # Image size
    height, width = 1852, 3076
    image_files = glob.glob(os.path.join(background_path, '*.png')) + \
                  glob.glob(os.path.join(background_path, '*.jpg')) + \
                  glob.glob(os.path.join(background_path, '*.jpeg'))
    if not image_files:
        raise ValueError(f"No image files found in the specified background path: {background_path}")

    # Randomly choose one image from the background folder
    chosen_image_path = random.choice(image_files)
    background_image = io.imread(chosen_image_path)
    print(f"Chosen background image: {chosen_image_path}")
    if background_image.shape[2] == 4:  # Check if the image has an alpha channel
        background_image = background_image[..., :3]  # Keep only RGB channels

    # Resize the background image to the desired canvas size
    background_image = resize(background_image, (1000, 1000, 3), anti_aliasing=True)
    background_image = tile_image_to_size(background_image, height, width)
    background_image = (background_image * 255).astype(np.uint8)

    return background_image

def tile_image_to_size(image, target_height, target_width):
    h, w, c = image.shape
    reps_y = -(-target_height // h)  # Ceiling division
    reps_x = -(-target_width // w)
    tiled = np.tile(image, (reps_y, reps_x, 1))
    return tiled[:target_height, :target_width, :]

# %%
