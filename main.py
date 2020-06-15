import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import tempfile

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import compression
import models
import datasets

architecture = 'alexnet'
dataset = 'cifar10'

batch_size = 32
epochs = 100


# Get Dataset
if dataset == 'mnist':
    img_rows, img_cols, img_channels = 28, 28, 1
    num_classes = 10
    (x_train, y_train), (x_test, y_test), input_shape = datasets.get_mnist((img_rows, img_cols, img_channels))

elif dataset == 'cifar10':
    img_rows, img_cols, img_channels = 32, 32, 3
    num_classes = 10
    (x_train, y_train), (x_test, y_test), input_shape = datasets.get_cifar10((img_rows, img_cols, img_channels))

elif dataset == 'cifar100':
    img_rows, img_cols, img_channels = 32, 32, 3
    num_classes = 100
    (x_train, y_train), (x_test, y_test), input_shape = datasets.get_cifar100((img_rows, img_cols, img_channels))


# Get model
if architecture == 'alexnet':
    model = models.AlexNetModel(input_shape, num_classes)
elif architecture == 'vgg19':
    model = models.VGG19Model(input_shape, num_classes)

tf.keras.utils.plot_model(model, to_file='{0}/{0}_{1}_original.png'.format(architecture, dataset))

logdir = tempfile.mkdtemp()
print('Writing training logs to ' + logdir)

keras_file = '{0}/{0}_{1}_temp.h5'.format(architecture, dataset)
tb = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=round(epochs/4))
mc = ModelCheckpoint(keras_file, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

callbacks = [tb, es, mc]

opt = tf.keras.optimizers.Adam()

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=opt,
    metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('Saving model to: ', keras_file.replace('temp', 'original'))
tf.keras.models.save_model(model, keras_file.replace('temp', 'original'), include_optimizer=False)

