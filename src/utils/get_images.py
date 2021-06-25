import numpy as np
from image_slicer import slice
import cv2


def get_figures_images(path: str):
    slice(path, col=8, row=8)


def get_figures_arrays(path: str) -> list:
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
