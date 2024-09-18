from PIL import Image
import numpy as np


def convert_to_index_image(image):
    pixel_mapping = {
        0: 0,  # TP
        1: 1,  # TN
        2: 2,  # FP
        3: 3   # FN
    }
    color_mapping = {
        0: (0, 0, 0),        # black
        1: (255, 255, 255),  # white
        2: (0, 255, 0),      # green
        3: (255, 0, 0),      # red
    }

    width, height = image.size
    index_image = Image.new("P", (width, height))

    pixels = np.array(image)
    index_pixels = np.zeros_like(pixels, dtype=np.uint8)
    for pixel_value, index_value in pixel_mapping.items():
        index_pixels[pixels == pixel_value] = index_value
    index_image.putdata(index_pixels.ravel())

    palette = []
    for index_value, color_value in color_mapping.items():
        palette.extend(color_value)
    index_image.putpalette(palette)
    return index_image



