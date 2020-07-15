import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from compression import prune_weights, check_sparsity
import models
import os
from config import config as cfg
import datasets
import argparse
from framework_lra import lra_framework


def train(model:tf.keras.models) -> tf.keras.models:
    logdir = tempfile.mkdtemp()
    print('Writing training logs to ' + logdir)
    keras_file = '{0}/{0}_{1}_temp.h5'.format(args.architecture, args.dataset)
    tb = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=round(cfg.epochs / 4))
    mc = ModelCheckpoint(keras_file, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    learning_rate_scheduler = LearningRateScheduler(schedule=cfg.learning_rate_scheduler)
    callbacks = [tb, es, mc, learning_rate_scheduler]
    opt = tf.keras.optimizers.Adam()
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=opt,
        metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=cfg.batch_size,
              epochs=cfg.epochs,
              verbose=1,
              callbacks=callbacks,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print('Saving model to: ', keras_file.replace('temp', 'original'))
    tf.keras.models.save_model(model, keras_file.replace('temp', 'original'), include_optimizer=False)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
          '--arch', type=str, default='alexnet', help='Type of model: alexnet, vgg19, mnist ')
    parser.add_argument(
          '--dataset', type=str, default='cifar10', help='dataset name: mnist, cifar10 or cifar100 ')
    parser.add_argument(
          '--path', type=str, help='path to model ')
    parser.add_argument(
          '--train', default=False, help='Initial train')
    parser.add_argument(
          '--lra', default=True, help='Low Rank approximation')
    parser.add_argument(
          '--prune', default=False, help='preform pruning before lra')
    parser.add_argument(
        '--plot_model', default=False, help='plot model')
    parser.add_argument(
        '--lra_algo', type=str, default='tsvd', help='lra algorithm')
    args = parser.parse_args()
    # Get Dataset
    if args.dataset == 'mnist':
        img_rows, img_cols, img_channels = 28, 28, 1
        num_classes = 10
        (x_train, y_train), (x_test, y_test), input_shape = datasets.get_mnist((img_rows, img_cols, img_channels))
    elif args.dataset == 'cifar10':
        img_rows, img_cols, img_channels = 32, 32, 3
        num_classes = 10
        (x_train, y_train), (x_test, y_test), input_shape = datasets.get_cifar10((img_rows, img_cols, img_channels))
    else:
        img_rows, img_cols, img_channels = 32, 32, 3
        num_classes = 100
        (x_train, y_train), (x_test, y_test), input_shape = datasets.get_cifar100((img_rows, img_cols, img_channels))
    # Get model
    if args.arch == 'alexnet':
        model = models.AlexNetModel(input_shape, num_classes)
    elif args.arch == 'vgg19':
        model = models.VGG19Model(input_shape, num_classes)
    else:
        model = models.MnistDense(input_shape, num_classes)
    model_dir = r'{}'.format(args.arch)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.plot_model:  # plot model
        tf.keras.utils.plot_model(model, to_file='{0}/{0}_{1}_original.png'.format(args.arch, args.dataset))
    if args.train:  # train model
        model = train(model)
    elif args.path is not None:
        model = load_model(args.path)
    else:
        model = load_model('{0}/{0}_{1}_original.h5'.format(args.arch, args.dataset))
    opt = tf.keras.optimizers.Adam()
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=opt,
        metrics=['accuracy'])
    if args.prune:  # prune model
        # check_sparsity(model)
        # model = prune_weights(model)
        check_sparsity(model)
        args.arch += '_pruned'  # TODO add ratio of pruning for various models
    if args.lra:  # lra model using lra algo and our proposed framework
        model = lra_framework(model, lra_algorithm=args.lra_algo, x_train=x_train, x_test=x_test, y_test=y_test,
                              dataset=args.dataset, model_name=args.arch)
