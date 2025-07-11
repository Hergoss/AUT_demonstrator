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
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon


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
            cy, cx = np.random.randint(0, height-1), np.random.randint(0, width-1)
            ry, rx = np.random.randint(30, 200), np.random.randint(30, 200)
            angle = np.random.uniform(0, 2*np.pi)
            y, x = np.ogrid[:height, :width]
            ellipse_mask = (((((x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)) / rx) ** 2 +
                            (((x - cx) * np.sin(angle) - (y - cy) * np.cos(angle)) / ry) ** 2) <= 1)
            image[ellipse_mask] = color

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
    """    Create a background image with random shapes.
    This function generates a background image with random shapes such as rectangles, circles, ellipses, and polygons.
    Args:
        num_shapes (int): Number of random shapes (set this variable as needed)
    
    Returns:
        np.ndarray: The generated background image.
    """

    # Image size
    height, width = 1852, 3076

    # Create almost white (light gray) background
    background_int = np.random.randint(1, 255)
    background_color = [background_int, background_int, background_int]  # gray background
    image = np.ones((height, width, 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)

    for _ in range(num_shapes):
        shape_type = np.random.choice(['rectangle', 'circle', 'ellipse', 'polygon'])
        color = [np.random.randint(180, 220) for _ in range(3)]  # random gray

        if shape_type == 'rectangle':
            max_width = 400
            max_height = 400

            # Choose top-left point ensuring there's space for the rectangle
            x1 = np.random.randint(0, width - max_width)
            y1 = np.random.randint(0, height - max_height)

            # Add random width/height up to max
            rect_width = np.random.randint(50, max_width)
            rect_height = np.random.randint(50, max_height)

            x2 = x1 + rect_width
            y2 = y1 + rect_height

            # Ensure x2, y2 still within bounds
            x2 = min(x2, width - 1)
            y2 = min(y2, height - 1)

            # Draw rectangle
            rr, cc = polygon([y1, y1, y2, y2], [x1, x2, x2, x1])
            rr = np.clip(rr, 0, height - 1)
            cc = np.clip(cc, 0, width - 1)
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
            cy, cx = np.random.randint(0, height-1), np.random.randint(0, width-1)
            ry, rx = np.random.randint(30, 200), np.random.randint(30, 200)
            angle = np.random.uniform(0, 2*np.pi)
            y, x = np.ogrid[:height, :width]
            ellipse_mask = (((((x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)) / rx) ** 2 +
                            (((x - cx) * np.sin(angle) - (y - cy) * np.cos(angle)) / ry) ** 2) <= 1)
            image[ellipse_mask] = color

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

    # Resize the background image to the desired canvas size
    from skimage.transform import resize
    background_image = resize(background_image, (height, width, 3), anti_aliasing=True)
    background_image = (background_image * 255).astype(np.uint8)
    return background_image


# %%
