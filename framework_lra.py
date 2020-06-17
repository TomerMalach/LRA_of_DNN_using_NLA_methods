import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.models import save_model
from config import config as cfg
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import KLD
from scipy.special import softmax
from compression import lra_per_layer
import os

def kullback_leibler_divergence(y, x):
    return KLD(y, x).numpy()

def get_relevant_layers(model, type_of_relevant_layers=[Dense]):
    relevant_layers = []
    relevant_layers_index_in_model = []
    for i, layer in enumerate(model.layers):
        if any([isinstance(layer, type) for type in type_of_relevant_layers]):
            relevant_layers.append(layer)
            relevant_layers_index_in_model.append(i)
    return relevant_layers, relevant_layers_index_in_model

def get_layer_outputs(model, patches, layers=None):
    """
    Print internal layer outputs to files
    :param model: trained keras model
    :param patches: input samples
    :param layers: relevant layers, if None then all layer are relevant
    :return: layer_names, layer_outs
    """
    inp = model.input  # input placeholder
    if layers is None:
        outputs = [layer.output for layer in model.layers[1:]]
    else:
        outputs = [layer.get_output_at(-1) for layer in layers]
    functors = [K.function([inp, K.symbolic_learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outs = [func([patches, False])[0] for func in functors]
    layer_outs_dist = list(map(lambda x: softmax(x, axis=-1), layer_outs))
    return layer_outs_dist


def evaluate_kld_for_each_layer(model:Model, lra_model:Model, dataset, layers=None):
    # we dont need the labels because we only want P(y'|x;model) = P(y'|x; lra_model), we don't care if y' != y
    model_layer_Dense_output_dist = get_layer_outputs(model, patches=dataset, layers=layers)
    lra_model_layer_Dense_output_dist = get_layer_outputs(lra_model, patches=dataset, layers=layers)
    # true is the first argument in kl_div
    kld = list(map(kullback_leibler_divergence, model_layer_Dense_output_dist, lra_model_layer_Dense_output_dist))
    kld = np.mean(kld, axis=-1)
    return kld


def lra_framework(model: Model, lra_algorithm, x_train, x_test, y_test):
    samples = np.concatenate((x_train, x_test))
    samples = samples[:100]  # TODO for development only, delete when ready
    lra_model = model
    relevant_layers, relevant_layers_index_in_model = get_relevant_layers(model)
    for it in range(cfg.lra_iterations):
        kld_per_layer = evaluate_kld_for_each_layer(model, lra_model, samples, relevant_layers)
        min_kld_index = np.argmin(kld_per_layer)
        layer_with_min_kld = relevant_layers[min_kld_index]
        layer_index_in_model_with_min_kld = relevant_layers_index_in_model[min_kld_index]
        print("Low rank approximation of layer {} : {}".format(layer_index_in_model_with_min_kld, layer_with_min_kld.name))
        lra_model = lra_per_layer(lra_model, layer_index=layer_index_in_model_with_min_kld, algorithm=lra_algorithm)
        score = model.evaluate(x_test, y_test, verbose=1)
        print("Iteration {}/{}: \n \t test loss \t {} \n \t test accuracy {}".format(it, cfg.lra_iterations,  score[0],
                                                                                     score[1]))
        if score[1] < cfg.accuracy_tolerance:
            print("Iteration {}/{}: accuracy is lower than tolerance, ending lra".format(it, cfg.lra_iterations))
            break
    save_model_path = os.path.join(model.name, cfg.lra_model_path)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    print('Saving model to: ', save_model_path)
    save_model(model, save_model_path, include_optimizer=False)

