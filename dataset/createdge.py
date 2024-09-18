import os
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8) # Adjusted here

    return edgemap


def invert_border_pixels(image):
    width, height = image.size

    for x in range(width):
        for y in range(height):
            if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                pixel_value = 1 - image.getpixel((x, y))  # Adjusted here
                image.putpixel((x, y), pixel_value)

    return image


def convert_1_to_255(image):
    width, height = image.size

    for x in range(width):
        for y in range(height):
            if image.getpixel((x, y)) == 1:
                image.putpixel((x, y), 255)

    return image


# input_folder = ''
# output_folder = ''

# os.makedirs(output_folder, exist_ok=True)
# mask_files = os.listdir(input_folder)

# for mask_file in mask_files:
#     input_mask_path = os.path.join(input_folder, mask_file)
#     image = Image.open(input_mask_path)
#     mask = np.array(image)
#
#     num_classes = 2
#     onehot_mask = mask_to_onehot(mask, num_classes)
#
#     radius = 1
#     edge_map = onehot_to_binary_edges(onehot_mask, radius, num_classes)
#
#     edge_image_with_inverted_border = invert_border_pixels(Image.fromarray(edge_map[0]))
#     final_image = convert_1_to_255(edge_image_with_inverted_border)
#
#     output_mask_file = os.path.splitext(mask_file)[0] + '.png'
#     output_edge_path = os.path.join(output_folder, output_mask_file)
#     final_image.save(output_edge_path)