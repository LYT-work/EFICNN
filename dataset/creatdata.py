import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def non_overlapping_crops(input_folderA, input_folderB, input_folderC, output_folderA, output_folderB, output_folderC, crop_size):
    file_list = os.listdir(input_folderA)

    if not os.path.exists(output_folderA):
        os.makedirs(output_folderA)
    if not os.path.exists(output_folderB):
        os.makedirs(output_folderB)
    if not os.path.exists(output_folderC):
        os.makedirs(output_folderC)

    for i in range(len(file_list)):
        file = file_list[i]

        input_fileA = os.path.join(input_folderA, file)
        input_fileB = os.path.join(input_folderB, file)
        input_fileC = os.path.join(input_folderC, file)

        input_imageA = Image.open(input_fileA)
        input_imageB = Image.open(input_fileB)
        input_imageC = Image.open(input_fileC)

        width, height = input_imageA.size

        num_rows = height // crop_size
        num_cols = width // crop_size

        for row in range(num_rows):
            for col in range(num_cols):
                x = col * crop_size
                y = row * crop_size

                input_dataA = input_imageA.crop((x, y, x + crop_size, y + crop_size))
                input_dataB = input_imageB.crop((x, y, x + crop_size, y + crop_size))
                input_dataC = input_imageC.crop((x, y, x + crop_size, y + crop_size))

                output_fileA = os.path.join(output_folderA, f"{i}_{row}_{col}.png")
                output_fileB = os.path.join(output_folderB, f"{i}_{row}_{col}.png")
                output_fileC = os.path.join(output_folderC, f"{i}_{row}_{col}.png")

                input_dataA.save(output_fileA)
                input_dataB.save(output_fileB)
                input_dataC.save(output_fileC)


def overlapping_crops(input_folderA, input_folderB, input_folderC, output_folderA, output_folderB, output_folderC, crop_size, overlap_ratio=0.5):
    file_list = os.listdir(input_folderA)

    if not os.path.exists(output_folderA):
        os.makedirs(output_folderA)
    if not os.path.exists(output_folderB):
        os.makedirs(output_folderB)
    if not os.path.exists(output_folderC):
        os.makedirs(output_folderC)

    for i in range(len(file_list)):
        file = file_list[i]

        input_fileA = os.path.join(input_folderA, file)
        input_fileB = os.path.join(input_folderB, file)
        input_fileC = os.path.join(input_folderC, file)

        input_imageA = Image.open(input_fileA)
        input_imageB = Image.open(input_fileB)
        input_imageC = Image.open(input_fileC)

        width, height = input_imageA.size

        overlap_pixels = int(crop_size * overlap_ratio)
        stride = crop_size - overlap_pixels

        num_rows = (height - overlap_pixels) // stride
        num_cols = (width - overlap_pixels) // stride

        for row in range(num_rows):
            for col in range(num_cols):
                x = col * stride
                y = row * stride

                input_dataA = input_imageA.crop((x, y, x + crop_size, y + crop_size))
                input_dataB = input_imageB.crop((x, y, x + crop_size, y + crop_size))
                input_dataC = input_imageC.crop((x, y, x + crop_size, y + crop_size))

                output_fileA = os.path.join(output_folderA, f"{i}_{row}_{col}.png")
                output_fileB = os.path.join(output_folderB, f"{i}_{row}_{col}.png")
                output_fileC = os.path.join(output_folderC, f"{i}_{row}_{col}.png")

                input_dataA.save(output_fileA)
                input_dataB.save(output_fileB)
                input_dataC.save(output_fileC)


input_folderA = ''
input_folderB = ''
input_folderC = ''

output_folderA = ''
output_folderB = ''
output_folderC = ''

crop_size = 256
non_overlapping_crops(input_folderA, input_folderB, input_folderC, output_folderA, output_folderB, output_folderC, crop_size)
