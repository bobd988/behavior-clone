import errno
import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.stats import bernoulli


# Some useful constants
DRIVING_LOG_FILE = '../data/driving_log.csv'
IMG_PATH = '../data/'
STEERING_COEFFICIENT = 0.229
DEBUGING_FLAG = False


def crop_image(image, top_percent, bottom_percent):

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]


def resize(image, new_dim):

    return scipy.misc.imresize(image, new_dim)

def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop_image(image,0.35, 0.1)
    image = resize(image,new_dim=(66, 200))
    #image = rgb2yuv(image)
    return image

def random_flip(image, steering_angle, flipping_prob=0.5):

    prob_flip = np.random.randint(2)
    if prob_flip == 0:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
        if DEBUGING_FLAG:
            # Show image if Debug Flag is enabled
            scipy.misc.imsave('flip.jpg', image)
    return image, steering_angle

def random_gamma(image):

    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def random_shear(image, steering_angle, shear_range=200):

    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle


def augment_image(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(66, 200), do_shear_prob=0.9):
    if DEBUGING_FLAG:
        # Show image if Debug Flag is enabled
        scipy.misc.imsave('origin.jpg', image)

    head = bernoulli.rvs(do_shear_prob)
    if head == 1:
        image, steering_angle = random_shear(image, steering_angle)
        if DEBUGING_FLAG:
            # Show image if Debug Flag is enabled
            scipy.misc.imsave('shear.jpg', image)

    image = crop_image(image, top_crop_percent, bottom_crop_percent)
    if DEBUGING_FLAG:
        # Show image if Debug Flag is enabled
        scipy.misc.imsave('crop.jpg', image)

    image, steering_angle = random_flip(image, steering_angle)

    image = random_gamma(image)
    if DEBUGING_FLAG:
        #Show image if Debug Flag is enabled
        scipy.misc.imsave('gamma.jpg', image)

    image = resize(image, resize_dim)
    if DEBUGING_FLAG:
        # Show image if Debug Flag is enabled
        scipy.misc.imsave('resize.jpg', image)

    return image, steering_angle


def get_batch_files(batch_size=64):

    data = pd.read_csv(DRIVING_LOG_FILE)
    num_of_img = len(data)
    rnd_indices = np.random.randint(0, num_of_img, batch_size)

    image_files_and_angles = []
    for index in rnd_indices:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + STEERING_COEFFICIENT
            image_files_and_angles.append((img, angle))

        elif rnd_image == 1:
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            image_files_and_angles.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - STEERING_COEFFICIENT
            image_files_and_angles.append((img, angle))

    return image_files_and_angles


def generator_training(batch_size=64):

    while True:
        X_batch = []
        y_batch = []
        images = get_batch_files(batch_size)
        for img_file, angle in images:
            raw_image = plt.imread(IMG_PATH + img_file)
            raw_angle = angle
            new_image, new_angle = augment_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        yield np.array(X_batch), np.array(y_batch)

def generator_validation(batch_size=64):

    while True:
        X_batch = []
        y_batch = []
        images = get_batch_files(batch_size)
        for img_file, angle in images:
            raw_image = plt.imread(IMG_PATH + img_file)
            raw_angle = angle
            #new_image, new_angle = generate_new_image(raw_image, raw_angle)
            new_image = preprocess(raw_image)
            new_angle = raw_angle

            X_batch.append(new_image)
            y_batch.append(new_angle)

        yield np.array(X_batch), np.array(y_batch)


def save_model(model, model_name='model.json', weights_name='model.h5'):

    delete_file(model_name)
    delete_file(weights_name)
    json_string = model.to_json()
    with open(model_name, 'w') as outfile:
        json.dump(json_string, outfile)

    model.save_weights(weights_name)


def delete_file(file):

    try:
        os.remove(file)

    except OSError as error:
        if error.errno != errno.ENOENT:
            raise


