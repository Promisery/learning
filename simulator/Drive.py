# <p>In the next video, I introduce Flask &amp; Socket.io to establish bi-directional client-server communication. Ultimately, this will be done to connect our model to the simulation. </p><p> That being said, the content in the next two videos is very technical, and not very relevant to deep learning. I would still highly recommend following along the two videos so that you don't get lost. </p><p>Otherwise, if you choose to skip the two videos, then that's fine. You'll have to simply copy the code below into an atom project, drag your model into the same project, make the appropriate installations and run the code to establish the connection. Then run your simulation. </p><p><br></p><pre class="prettyprint linenums">import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import socketio
import cv2
import math

sio = socketio.Server()

app = Flask(__name__)  # '__main__'
speed_limit = 25


def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    # steering_angle = math.sin(steering_angle)
    throttle = 1.0 - speed / speed_limit
    # throttle = 0.3
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
# </pre>
