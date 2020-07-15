import numpy as np
from numpy.linalg import cond
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import save_model, clone_model
from config import config as cfg
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.losses import KLD
from scipy.special import softmax
from functools import reduce
from compression import lra_per_layer
import os
from plot_utils import *
from logger_utils import get_logger


def evaluate_condition_number_for_each_layer(model:Model):
    cond_num = []
    for layer in model.layers:
        weights = layer.get_weights()
        for weight in weights:
            if len(weight.shape) > 2:
                weights2d = weight.reshape([weight.shape[-1], -1])
            elif len(weight.shape) == 1:
                continue
            else:
                weights2d = weight
            cond_num.append(cond(weights2d))
    return cond_num

def get_initial_number_of_params(layers):
    num_of_params = 0
    for layer in layers:
        weights = layer.get_weights()
        for weight in weights:
            num_of_params += reduce(lambda x1, x2: x1 * x2, np.shape(weight))

    return num_of_params

def kullback_leibler_divergence(y, x):
    return KLD(y, x).numpy()

def get_relevant_layers(model, type_of_relevant_layers=[Dense,Conv2D]):
    relevant_layers = []
    relevant_layers_index_in_model = []
    for i, layer in enumerate(model.layers):
        if any([isinstance(layer, type) for type in type_of_relevant_layers]):
            relevant_layers.append(layer)
            relevant_layers_index_in_model.append(i)
    return relevant_layers, relevant_layers_index_in_model

def get_layer_outputs(model, patches):
    """
    Print internal layer outputs to files
    :param model: trained keras model
    :param patches: input samples
    :param layers: relevant layers, if None then all layer are relevant
    :return: layer_outs probability dist
    """
    inp = model.input  # input placeholder
    layers, _ = get_relevant_layers(model=model)
    if layers is None:
        outputs = [layer.output for layer in model.layers[1:]]
    else:
        outputs = [layer.output for layer in layers]
    functors = [K.function([inp, K.symbolic_learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outs = [func([patches, False])[0] for func in functors]
    layer_outs_dist = list(map(lambda x: softmax(x, axis=-1), layer_outs))
    return layer_outs_dist


def evaluate_kld_for_each_layer(model:Model, lra_model:Model, dataset):
    # we dont need the labels because we only want P(y'|x;model) = P(y'|x; lra_model), we don't care if y' != y

    model_layer_Dense_output_dist = get_layer_outputs(model, patches=dataset)
    lra_model_layer_Dense_output_dist = get_layer_outputs(lra_model, patches=dataset)
    # true is the first argument in kl_div
    kld = list(map(kullback_leibler_divergence, model_layer_Dense_output_dist, lra_model_layer_Dense_output_dist))
    kld = np.mean(kld, axis=-1)
    kld[kld < 0] = 0
    return kld




def evaluate_kld_for_last_layer(model:Model, lra_model:Model, dataset):
    # we dont need the labels because we only want P(y'|x;model) = P(y'|x; lra_model), we don't care if y' != y
    sampled_dataset = dataset[np.random.choice(range(len(dataset)), size=500, replace=False), :]
    model_layer_Dense_output_dist = model.predict(sampled_dataset)
    lra_model_layer_Dense_output_dist = lra_model.predict(sampled_dataset)
    # true is the first argument in kl_div
    kld = kullback_leibler_divergence(y=model_layer_Dense_output_dist, x=lra_model_layer_Dense_output_dist)
    # for kld_l in kld:
    #     if any([kld_i < 0 for kld_i in kld_l]):
    #         a  = 5
    kld = np.mean(kld, axis=-1)
    kld = 0 if kld < 0 else kld
    return kld

def lra_framework(model: Model, lra_algorithm, x_train, x_test, y_test, dataset, model_name):
    scores = []
    logger_path = 'log_file_model_{}'.format(model_name)
    if os.path.exists(logger_path + ".log"):
        os.remove(logger_path + ".log")
    logger = get_logger(logger_path)
    samples = x_train  # np.concatennum_of_paramsate((x_train, x_test))

    initial_score = model.evaluate(x_test, y_test, verbose=0)


    score = np.copy(initial_score)

    score_to_plot = []
    compression_ratio_to_plot = []

    score_to_plot.append(score[-1])
    compression_ratio_to_plot.append(0)

    opt = tf.keras.optimizers.Adam()

    temp_model = clone_model(model)
    temp_model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=opt,
        metrics=['accuracy'])

    lra_model = clone_model(model)
    lra_model.set_weights(model.get_weights())
    lra_model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=opt,
        metrics=['accuracy'])

    relevant_layers, relevant_layers_index_in_model = get_relevant_layers(model)
    initial_num_of_params = get_initial_number_of_params(relevant_layers)
    print('\n\n')
    it = 0

    while (initial_score[1] - score[1] < cfg.accuracy_tolerance):
        klds = []
        print("Start of Iteration {0}:".format(it))
        logger.info("Start of Iteration {0}:".format(it))
        curr_num_of_params = 0
        for i, layer_index in enumerate(relevant_layers_index_in_model):
            temp_model.set_weights(lra_model.get_weights())

            temp_model, _, _, num_of_params_layer_i = lra_per_layer(temp_model, layer_index=layer_index,
                                                                    algorithm=lra_algorithm, update_memory=False)
            curr_num_of_params += num_of_params_layer_i
            # kld_per_layer = evaluate_kld_for_each_layer(model, temp_model, samples)
            kld_per_layer = evaluate_kld_for_last_layer(model, temp_model, samples)

            print('{} ({}): KLD per layer {:.4f}'.format(relevant_layers[i].name, layer_index, kld_per_layer))
            # klds.append(sum(kld_per_layer))
            klds.append(kld_per_layer)

        print("Compression ratio : {}. \n number of params: \n \t lra_model {:,} \n \t initial model {:,}"
              .format(1 - curr_num_of_params / initial_num_of_params, curr_num_of_params, initial_num_of_params))
        logger.info("Compression ratio : {}.  number of params:  \t lra_model {:,} \t initial model {:,}"
              .format(1 - curr_num_of_params / initial_num_of_params, curr_num_of_params, initial_num_of_params))

        if curr_num_of_params < initial_num_of_params:
            score_to_plot.append(score[-1])
            compression_ratio_to_plot.append(1 - curr_num_of_params / initial_num_of_params)  # so the graph will start from 0 and go to 1
        min_kld_index = np.argmin(klds)

        layer_with_min_kld = relevant_layers[min_kld_index]
        layer_index_in_model_with_min_kld = relevant_layers_index_in_model[min_kld_index]

        # print('---------------- Start Compression with {0} for layer {1}!) ----------------'.format(lra_algorithm,
        #                                                                                             layer_with_min_kld.name))
        lra_model, truncated, full_svs, _ = lra_per_layer(lra_model, layer_index=layer_index_in_model_with_min_kld,
                                                   algorithm=lra_algorithm, update_memory=True)
        print('Approximate {0} {1} using {2}/{3} singular values'.format(layer_with_min_kld.name,
                                                                         layer_index_in_model_with_min_kld,
                                                                         truncated, full_svs))
        logger.info('Approximate {0} {1} using {2}/{3} singular values'.format(layer_with_min_kld.name,
                                                                         layer_index_in_model_with_min_kld,
                                                                         truncated, full_svs))

        # print('---------------- Done Compression with {0} for layer {1}!) ----------------'.format(lra_algorithm,
        #                                                                                            layer_with_min_kld.name))

        score = lra_model.evaluate(x_test, y_test, verbose=0)

        scores.append(score)
        print("End of Iteration {}:\ntest loss = {:.3f}\ntest accuracy = {:.3f}\n\n\n".format(it,  score[0], score[1]))
        logger.info("End of Iteration {}:\ntest loss = {:.3f}\ntest accuracy = {:.3f}\n\n\n".format(it,  score[0], score[1]))
        it += 1

    if not os.path.exists(model_name):
        os.makedirs(model_name)
    save_model_path = os.path.join(model_name, '{0}_{1}_lra.h5'.format(model_name, dataset))
    print('Saving model to: ', save_model_path)
    logger.info('Saving model to:  {}'.format(save_model_path))
    save_model(lra_model, save_model_path, include_optimizer=False, save_format='h5')
    score_to_plot = np.asarray(score_to_plot)
    compression_ratio_to_plot = np.asarray(compression_ratio_to_plot)
    np.save(os.path.join(model_name, 'score_{}'.format(model_name)), score_to_plot)
    np.save(os.path.join(model_name, 'compression_{}'.format(model_name)), compression_ratio_to_plot)
    plot_score_versus_compression(save_dir=model_name, score_data=score_to_plot, compression_data=compression_ratio_to_plot)
