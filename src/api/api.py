import sys
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# sys.path.extend(['C:\\Users\\filif\\Desktop\\programowanko\\chess_position_detection'])

from src.models.model import get_model_predictions


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
def get_predictions(file: UploadFile = File(...)):
    return get_model_predictions(file)


if __name__ == '__main__':
    uvicorn.run(app)
