from tensorflow.keras.models import load_model
from src.utils.get_images import get_figures_arrays, is_figure
import matplotlib.pyplot as plt
import cv2
import numpy as np

FEN_dict = {'black_bishop': 'b', 'black_king': 'k', 'black_knight': 'n', 'black_pawn': 'p',
            'black_queen': 'q', 'black_rook': 'r', 'white_bishop': 'B', 'white_king': 'K',
            'white_knight': 'N', 'white_pawn': 'P', 'white_queen': 'Q', 'white_rook': 'R'}
model = load_model('C:\\Users\\filif\\Desktop\\programowanko\\chess_position_detection\\src\\models\\acc1.h5')


def get_model_predictions(encoded_image) -> str:
    images = get_figures_arrays(encoded_image, True)
    i = 1
    space = 0

    notation = ''
    for img in images:
        if is_figure(img):
            prediction = model.predict(img)
            if space is not 0:
                notation += str(space)
                space = 0
            notation += (FEN_dict[list(FEN_dict.keys())[np.argmax(prediction)]])
        else:
            space += 1
        if i % 8 is 0:
            if space is not 0:
                notation += str(space)
            notation += '/'
            space = 0
        i += 1
    return notation
