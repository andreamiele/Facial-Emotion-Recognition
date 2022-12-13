from threading import Thread
import cv2 as cv
import numpy as np
from keras.models import model_from_json
from PIL import ImageColor
from queue import Queue


class Predictor(Thread):

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue: Queue = queue
        self.emotions = []
        self.emotion_colors = []
        self.face_frames = []

        json_file = open('./models/CNN.json', 'r')
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json)
        # load weights into new model
        self.model.load_weights("./models/CNN.h5")
        print("Loaded model from disk")

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            image, faces = self.queue.get()
            try:
                if image is None:
                    break

                self._predict(image, faces)
            finally:
                self.queue.task_done()

    def _predict(self, image: cv.Mat, faces: list[tuple]):

        emotions = []
        emotion_colors = []
        face_frames = []

        for face in faces:
            (x, y, w, h) = face
            b = y + h
            r = x + w

            if w >= 48 and h >= 48:

                try:
                    face_image = image[y:b, x:r]

                    input_image = cv.resize(face_image, (48, 48))

                    input_image = input_image.reshape(
                        (1, 48, 48, 1)) / 255  # type: ignore

                    prediction = np.argmax(self.model.predict(input_image),
                                           axis=1)
                    # define emotions for square
                    emotion_label_to_text = {
                        0: 'anger',
                        1: 'disgust',
                        2: 'fear',
                        3: 'happiness',
                        4: 'sadness',
                        5: 'surprise',
                        6: 'neutral'
                    }
                    # define colors for square
                    colors = {
                        0: "#e74c3c",
                        1: "#16a085",
                        2: "#2c3e50",
                        3: "#f39c12",
                        4: "#3498db",
                        5: "#27ae60",
                        6: "#ffffff"
                    }

                    emotions.append(emotion_label_to_text.get(prediction[0]))

                    color = colors.get(prediction[0])
                    r, g, b = ImageColor.getcolor(color, "RGB")  # type: ignore
                    emotion_colors.append((b, g, r))

                    face_frames.append(face)

                except cv.error:
                    print("ERROR")

        self.emotions = emotions
        self.emotion_colors = emotion_colors
        self.face_frames = face_frames
