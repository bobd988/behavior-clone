import argparse
import base64
import json
from io import BytesIO
import helper
import eventlet.wsgi
import numpy as np
import socketio
import tensorflow as tf
from PIL import Image
from flask import Flask
from keras.models import model_from_json
import scipy.misc
import os
import shutil
from datetime import datetime

#tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED


def crop(image, top_percent, bottom_percent):
    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]


def resize(image, new_dim):

    return scipy.misc.imresize(image, new_dim)



@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]

        # The current throttle of the car
        throttle = data["throttle"]

        # The current speed of the car
        speed = data["speed"]

        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))

        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))

        try:
            image_array = np.asarray(image)
            #image_array = crop(image_array, 0.35, 0.1)
            #image_array = resize(image_array, new_dim=(66, 200))
            image_array = helper.preprocess(image_array)
            transformed_image_array = image_array[None, :, :, :]

            # This model currently assumes that the features of the model are just the images. Feel free to change this.
            steering_angle = 0.3

            #steering_angle = float(model.predict(transformed_image_array, batch_size=1))
            #global speed_limit
            #if speed > speed_limit:
            #    speed_limit = MIN_SPEED  # slow down
            #else:
            #    speed_limit = MAX_SPEED
            #throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2


            print('{:.5f}, {:.1f}'.format(steering_angle, throttle))

            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)
    else:

        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')

    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
