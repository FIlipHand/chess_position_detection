from tensorflow.keras.models import load_model
from src.utils.get_images import get_figures_arrays, is_figure
import matplotlib.pyplot as plt
import cv2
import numpy as np

FEN_dict = {'black_bishop': 'b', 'black_king': 'k', 'black_knight': 'n', 'black_pawn': 'p',
            'black_queen': 'q', 'black_rook': 'r', 'white_bishop': 'B', 'white_king': 'K',
            'white_knight': 'N', 'white_pawn': 'P', 'white_queen': 'Q', 'white_rook': 'R'}
model = load_model('C:\\Users\\filif\\Desktop\\programowanko\\chess_position_detection\\src\\models\\acc1.h5')


def get_model_predictions(encoded_image) -> dict:
    images = get_figures_arrays(encoded_image, True)
    preds = {}
    for i, img in enumerate(images):
        preds[i] = dict(zip(list(FEN_dict.keys()), [str(x) for x in list(model.predict(img).reshape(-1))]))
    return preds
