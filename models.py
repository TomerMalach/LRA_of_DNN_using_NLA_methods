import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ReLU, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.applications.vgg19 import VGG19

def AlexNetModel(input_shape, num_classes):
    # Original AlexNet
    x = Input(name='inputs', shape=input_shape, dtype='float32')

    o = Conv2D(96, 5, padding='same', strides=(2, 2), activation='relu')(x)
    o = BatchNormalization()(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(o)
    o = Dropout(0.25)(o)

    o = Conv2D(256, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(o)
    o = Dropout(0.25)(o)

    o = Conv2D(384, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)

    o = Conv2D(384, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)

    o = Conv2D(256, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(o)
    o = Dropout(0.25)(o)

    o = Flatten()(o)

    o = Dense(4096, activation='relu')(o)
    o = BatchNormalization()(o)
    o = Dropout(0.5)(o)

    o = Dense(4096, activation='relu')(o)
    o = BatchNormalization()(o)
    o = Dropout(0.5)(o)

    o = Dense(num_classes, activation='softmax')(o)

    model = Model(inputs=x, outputs=o)

    model.summary()

    return model


def VGG19Model(input_shape, num_classes):

    # base_model = VGG19(include_top=False, input_shape=input_shape, classes=num_classes)
    # # base_model.summary()
    # model = tf.keras.layers.Flatten()(base_model.layers[-1].output)
    # model = tf.keras.layers.Dense(4096, activation='relu')(model)
    # model = tf.keras.layers.Dense(4096, activation='relu')(model)
    # predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(model)
    # model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    x = Input(name='inputs', shape=input_shape, dtype='float32')

    o = Conv2D(64, 3, padding='same', strides=(1, 1), activation='relu')(x)
    o = BatchNormalization()(o)

    o = Conv2D(64, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(o)
    o = Dropout(0.25)(o)

    o = Conv2D(128, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)

    o = Conv2D(128, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(o)
    o = Dropout(0.25)(o)

    o = Conv2D(256, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)

    o = Conv2D(256, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(o)
    o = Dropout(0.25)(o)

    o = Conv2D(512, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)

    o = Conv2D(512, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)

    o = Conv2D(512, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)

    o = Conv2D(512, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(o)
    o = Dropout(0.25)(o)

    o = Conv2D(512, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)

    o = Conv2D(512, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)

    o = Conv2D(512, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)

    o = Conv2D(512, 3, padding='same', strides=(1, 1), activation='relu')(o)
    o = BatchNormalization()(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(o)
    o = Dropout(0.25)(o)

    o = Flatten()(o)

    o = Dense(4096, activation='relu')(o)
    o = BatchNormalization()(o)
    o = Dropout(0.5)(o)

    o = Dense(4096, activation='relu')(o)
    o = BatchNormalization()(o)
    o = Dropout(0.5)(o)

    o = Dense(num_classes, activation='softmax')(o)

    model = Model(inputs=x, outputs=o)

    model.summary()

    return model


def MnistDense(input_shape, num_classes):
    x = Input(name='inputs', shape=input_shape, dtype='float32')
    o = Flatten()(x)
    o = Dense(units=2048, name='hidden_layer', activation='sigmoid')(o)
    o = Dense(units=num_classes, name='output_layer',  activation='softmax')(o)
    model = Model(inputs=x, outputs=o, name='mnist')
    model.summary()
    return model