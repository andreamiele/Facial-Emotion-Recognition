# Import
import io
from queue import Queue
import cv2 as cv
import PySimpleGUIQt as sg
from Predictor import Predictor

# default sample frequency = 5
sample_frequency = 5


# define the window layout
sg.theme('DarkGreen1')

# Sample Frequency: 2/5/10
freq_widget = sg.Frame(
    title="Chose Your Sample Frequency",
    layout=[[
        sg.Radio(
            '2 frames',
            group_id='sample_size',
            enable_events=True,
            key='sample per 2',
            font='_ 12',
        ),
        sg.Radio(
            '5 frames',
            group_id='sample_size',
            enable_events=True,
            default=True,
            key='sample per 5',
            font='_ 12',
        ),
        sg.Radio(
            '10 frames',
            group_id='sample_size',
            enable_events=True,
            key='sample per 10',
            font='_ 12',
        ),
    ]],
    font='_ 14',
)

# Full Layout
layout = [
    [freq_widget],
    [sg.Image(filename='', key='CAM')],  # Camera Widget
]
window = sg.Window(
    'FER: Facial Emotion Recognition',
    layout,
    resizable=True,
    finalize=True,
)

face_cascade = cv.CascadeClassifier('face_default.xml')

buf = io.BytesIO()

cap = cv.VideoCapture(0)  # Setup the OpenCV capture device (webcam)

# Start queue
queue = Queue()
predictor_thread = Predictor(queue)
predictor_thread.start()

# ---------------------------------- Rendering -------------------------------

CAM_WINDOW_W = 700
frame_count = 1
while True:

    event, values = window.Read(timeout=20, timeout_key='timeout')
    if event in (sg.WIN_CLOSED, 'Quit'):
        break
    elif event == "sample per 2":
        sample_frequency = 2
    elif event == "sample per 5":
        sample_frequency = 5
    elif event == "sample per 10":
        sample_frequency = 10

    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if frame_count % sample_frequency == 0:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)

        # Sending image and faces to the predictor thread queue and continuing
        if queue.qsize() < 4:  # Making sure not overwhelm predictor == buffer
            queue.put((gray, faces))

    # Reading whatever the predictor thread has been able to predict
    emotions = predictor_thread.emotions
    emotion_colors = predictor_thread.emotion_colors
    face_frames = predictor_thread.face_frames

    # Displaying the predictions of the predictor thread == making a square around the face if detected
    for emotion, color, face_frame in zip(emotions, emotion_colors,
                                          face_frames):
        (x, y, w, h) = face_frame
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv.putText(
            frame,
            emotion,
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            1,
            cv.LINE_AA,
        )

    # Update camera window == print on the camera
    h, w, _ = frame.shape

    diff = ((CAM_WINDOW_W - 20) - w) / w

    new_w = CAM_WINDOW_W
    new_h = round(h + (h * diff))

    frame = cv.resize(frame, (new_w, new_h))  # type: ignore
    imgbytes = cv.imencode('.png', frame)[1].tobytes()
    window['CAM'].update(data=imgbytes)  # type: ignore

    if frame_count == 60:
        frame_count = 0

    frame_count += 1

window.close()
queue.put((None, None))
cap.release()
