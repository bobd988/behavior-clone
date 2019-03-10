import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import helper

STEERING_COEFFICIENT = 0.22
#col, row = 200,66
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
#bob tf.python.control_flow_ops = tf
number_of_epochs = 8
#bob number_of_samples_per_epoch = 20032
#number_of_validation_samples = 6400
number_of_samples_per_epoch = 20224
number_of_validation_samples = 6400
learning_rate = 1e-4
activation_relu = 'relu'



# the input is directly 66,200,3 . Not using resize to 64X64
# Our model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def build_model():

   model = Sequential()
   model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))

   # starts with five convolutional and maxpooling layers
   model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
   model.add(Activation(activation_relu))
   model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

   model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
   model.add(Activation(activation_relu))
   model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

   model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
   model.add(Activation(activation_relu))
   model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

   model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
   model.add(Activation(activation_relu))
   model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

   model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
   model.add(Activation(activation_relu))
   model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

   #model.add(Dropout(0.5))
   model.add(Flatten())

   # Next, five fully connected layers
   model.add(Dense(1164))
   model.add(Activation(activation_relu))

   model.add(Dense(100))
   model.add(Activation(activation_relu))

   model.add(Dense(50))
   model.add(Activation(activation_relu))

   model.add(Dense(10))
   model.add(Activation(activation_relu))

   model.add(Dense(1))

   model.summary()

   return model



def train_model(model):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    model.compile(optimizer=Adam(learning_rate), loss="mse", )

    # create generators for training and validation
    train_generator = helper.generator_training()
    validation_generator = helper.generator_validation()

    history = model.fit_generator(train_generator,
                                  samples_per_epoch=number_of_samples_per_epoch,
                                  nb_epoch=number_of_epochs,
                                  validation_data=validation_generator,
                                  nb_val_samples=number_of_validation_samples,
                                  callbacks=[checkpoint],
                                  verbose=1)


def main():
    model = build_model()
    train_model(model)
    helper.save_model(model)


if __name__ == '__main__':
    main()
