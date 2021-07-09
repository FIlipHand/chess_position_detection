import numpy as np
from image_slicer import slice, validate_image, validate_image_col_row, calc_columns_rows, Tile
from math import sqrt, ceil, floor
import base64
import cv2
import io
from PIL import Image


def get_figures_images(path: str):
    slice(path, col=8, row=8)


def get_figures_arrays(path: str, encoded: bool = False) -> list:
    if encoded:
        images = list(encoded_slice(path, row=8, col=8))
    else:
        images = list(slice(path, row=8, col=8, save=False))
    for i, img in enumerate(images):
        images[i] = np.array(img.image).dot(np.array([0.2989, 0.5870, 0.1140]))
        images[i] = cv2.resize(images[i], (50, 50))
        images[i] = images[i].reshape(1, images[i].shape[0], images[i].shape[1], 1)
    return images


def is_figure(image: np.array) -> bool:
    # means = []
    # for column in range(image.shape[1]):
    #     means.append(np.mean(image[:, column]))
    # print(np.std(means))
    if np.std(image) > 40:
        return True
    else:
        return False


#TODO tego moze raczej nie byc jak bede madry
def encoded_slice(encoded_image, number_tiles=None, col=None, row=None):
    im = Image.open(encoded_image.file)
    im_w, im_h = im.size

    columns = 0
    rows = 0
    if number_tiles:
        validate_image(im, number_tiles)
        columns, rows = calc_columns_rows(number_tiles)
    else:
        validate_image_col_row(im, col, row)
        columns = col
        rows = row

    tile_w, tile_h = int(floor(im_w / columns)), int(floor(im_h / rows))

    tiles = []
    number = 1
    for pos_y in range(0, im_h - rows, tile_h):  # -rows for rounding error.
        for pos_x in range(0, im_w - columns, tile_w):  # as above.
            area = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            image = im.crop(area)
            position = (int(floor(pos_x / tile_w)) + 1, int(floor(pos_y / tile_h)) + 1)
            coords = (pos_x, pos_y)
            tile = Tile(image, number, position, coords)
            tiles.append(tile)
            number += 1
    return tuple(tiles)
