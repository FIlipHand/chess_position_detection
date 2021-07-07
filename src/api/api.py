import sys
from fastapi import FastAPI, File
import uvicorn

# sys.path.extend(['C:\\Users\\filif\\Desktop\\programowanko\\chess_position_detection'])

from src.models.model import get_model_predictions


app = FastAPI()


@app.post("/predict")
def get_predictions(file: bytes = File(...)):
    return get_model_predictions(file)


if __name__ == '__main__':
    uvicorn.run(app)
