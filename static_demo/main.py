import base64
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, File, UploadFile
from keras.preprocessing.image import img_to_array
from PIL import Image
import uvicorn
import cv2 as cv
import numpy as np

import predict_image

app = FastAPI(root_path="/")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static/templates")

origins = ["http://localhost:8855", "http://localhost:8855/upload_file", "*"]


@app.get("/", response_class=HTMLResponse)
async def display_home(request: Request):
    return templates.TemplateResponse('home.html', {'request': request})


@app.post("/upload_file")
async def upload_file(user_image: UploadFile = File(...)):
    image = Image.open(user_image.file)

    np_image = img_to_array(image)
    np_image = np.array(np_image, dtype='uint8')
    color = cv.cvtColor(np_image, cv.IMREAD_ANYCOLOR)
    gray = cv.cvtColor(np_image, cv.COLOR_BGR2GRAY)

    face_cascade = cv.CascadeClassifier('face_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)

    emotions = []
    face_count = 0
    for face in faces:
        (x, y, w, h) = face
        face_count += 1
        cv.rectangle(color, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(color, f"Face {face_count}", (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        emotions.append(predict_image.run(gray, face))

    image_bytes = cv.imencode('.png', color)[1]
    encoded_image_string = base64.b64encode(image_bytes)

    return {"image": encoded_image_string, "emotions": emotions}


if __name__ == "__main__":
    uvicorn.run(
        app,  # type: ignore
        host="localhost",
        port=8855,
    )
