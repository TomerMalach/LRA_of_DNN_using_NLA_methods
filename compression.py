from tensorflow.keras.models import Model
import numpy as np
import sys
from matplotlib import pyplot as plt

def check_sparsity(model):

    conv_zero_params = []
    conv_total_params = []

    fc_zero_params = []
    fc_total_params = []

    for i in range(0, len(model.layers)):
        if "Conv" not in type(model.layers[i]).__name__ and "Dense" not in type(model.layers[i]).__name__:
            continue

        weights = model.layers[i].get_weights()

        for j in range(0, len(weights)):

            if "Conv" in type(model.layers[i]).__name__:
                conv_zero_params.append(weights[j].size - np.count_nonzero(weights[j]))
                conv_total_params.append(weights[j].size)

            else:
                fc_zero_params.append(weights[j].size - np.count_nonzero(weights[j]))
                fc_total_params.append(weights[j].size)

    print('Conv Sparsity: {0}'.format(sum(conv_zero_params) / (sum(conv_total_params) + 1)))
    print('FC Sparsity: {0}'.format(sum(fc_zero_params) / (sum(fc_total_params) + 1)))
    print('Total Sparsity: {0}'.format((sum(conv_zero_params) + sum(fc_zero_params)) /
                                       (sum(conv_total_params) + sum(fc_total_params) + 1)))
    print('Total Parameters: {0:,}'.format(sum(conv_total_params) + sum(fc_total_params)))


def prune_weights(model, conv_pruning_percent=0.3, fc_pruning_percent=0.7, per_channel=False,
                  fix_conv_pruning=False, fix_fc_pruning=False)->Model:

    # Prune the weights

    conv_zero_params = []
    conv_total_params = []

    fc_zero_params = []
    fc_total_params = []

    positive_pruned_sum = 0
    negative_pruned_sum = 0
    positive_survived_sum = 0
    negative_survived_sum = 0

    print('---------------- ' + ('Start Pruning Per Channel!' if per_channel else 'Start Pruning!') +
          ' ----------------\n')

    for i in range(0, len(model.layers)):

        if "Conv" not in type(model.layers[i]).__name__ and "Dense" not in type(model.layers[i]).__name__:
            continue

        weights = model.layers[i].get_weights()

        for j in range(0, len(weights)):

            if len(weights[j].shape) == 1:  # Skip biases pruning
                continue

            if "Conv" in type(model.layers[i]).__name__ and conv_pruning_percent > 0.0:

                if per_channel and weights[j].shape[0] * weights[j].shape[1] > 1:

                    for cin in range(0, weights[j].shape[2]):

                        weights_sorted = np.sort(np.abs(weights[j][:, :, cin, :]).flatten())
                        weight_thr = weights_sorted[round(weights_sorted.size * conv_pruning_percent)]

                        if fix_conv_pruning:

                            for cout in range(0, weights[j].shape[3]):

                                kernel = weights[j][:, :, cin, cout]

                                positive_pruned_sum = np.sum(kernel[np.logical_and(kernel < weight_thr, kernel > 0)])
                                negative_pruned_sum = np.sum(kernel[np.logical_and(kernel > -weight_thr, kernel < 0)])
                                positive_survived_sum = np.sum(kernel[kernel >= weight_thr])
                                negative_survived_sum = np.sum(kernel[kernel <= -weight_thr])

                                kernel[np.abs(kernel) < weight_thr] = 0.0

                                kernel[kernel > 0] += positive_pruned_sum * kernel[kernel > 0] / positive_survived_sum
                                kernel[kernel < 0] += negative_pruned_sum * kernel[kernel < 0] / negative_survived_sum
                        else:
                            kernel = weights[j][:, :, cin, :]
                            kernel[np.abs(kernel) < weight_thr] = 0.0

                else:

                    weights_sorted = np.sort(np.abs(weights[j]).flatten())
                    weight_thr = weights_sorted[round(weights_sorted.size * conv_pruning_percent)]

                    if fix_conv_pruning and weights[j].shape[0]*weights[j].shape[1] > 1:

                        for cin in range(0, weights[j].shape[2]):
                            for cout in range(0, weights[j].shape[3]):

                                kernel = weights[j][:, :, cin, cout]

                                positive_pruned_sum = np.sum(kernel[np.logical_and(kernel < weight_thr, kernel > 0)])
                                negative_pruned_sum = np.sum(kernel[np.logical_and(kernel > -weight_thr, kernel < 0)])
                                positive_survived_sum = np.sum(kernel[kernel >= weight_thr])
                                negative_survived_sum = np.sum(kernel[kernel <= -weight_thr])

                                kernel[np.abs(kernel) < weight_thr] = 0.0

                                kernel[kernel > 0] += positive_pruned_sum * kernel[kernel > 0] / positive_survived_sum
                                kernel[kernel < 0] += negative_pruned_sum * kernel[kernel < 0] / negative_survived_sum

                    else:
                        kernel = weights[j]
                        kernel[np.abs(kernel) < weight_thr] = 0.0

                conv_zero_params.append(weights[j].size - np.count_nonzero(weights[j]))
                conv_total_params.append(weights[j].size)

            elif "Dense" in type(model.layers[i]).__name__ and fc_pruning_percent > 0.0:  # FC layer

                weights_sorted = np.sort(np.abs(weights[j].flatten()))
                weight_thr = weights_sorted[int(weights_sorted.size * fc_pruning_percent)]

                for neuron_index in range(0, weights[j].shape[1]):

                    neuron = weights[j][:, neuron_index]

                    if fix_fc_pruning:
                        positive_pruned_sum = np.sum(neuron[np.logical_and(neuron < weight_thr, neuron > 0)])
                        negative_pruned_sum = np.sum(neuron[np.logical_and(neuron > -weight_thr, neuron < 0)])
                        positive_survived_sum = np.sum(neuron[neuron >= weight_thr])
                        negative_survived_sum = np.sum(neuron[neuron <= -weight_thr])

                    neuron[np.abs(neuron) < weight_thr] = 0.0

                    if fix_fc_pruning:
                        neuron[neuron > 0] += positive_pruned_sum * neuron[neuron > 0] / positive_survived_sum
                        neuron[neuron < 0] += negative_pruned_sum * neuron[neuron < 0] / negative_survived_sum

                fc_zero_params.append(weights[j].size - np.count_nonzero(weights[j]))
                fc_total_params.append(weights[j].size)

        model.layers[i].set_weights(weights)

        if round(100 * i / len(model.layers)) > round(100 * (i - 1) / len(model.layers)):
            sys.stdout.write('\r' + 'Pruning process: {0}%'.format(round(100 * i / len(model.layers))))
            sys.stdout.flush()

    sys.stdout.write('\r' + 'Pruning process: 100%\n')
    sys.stdout.flush()

    print('\nConv Sparsity: {0}'.format(sum(conv_zero_params) / (sum(conv_total_params) + 1)))
    print('FC Sparsity: {0}'.format(sum(fc_zero_params)/(sum(fc_total_params) + 1)))
    print('Total Sparsity: {0}'.format((sum(conv_zero_params) + sum(fc_zero_params))/
          (sum(conv_total_params) + sum(fc_total_params) + 1)))
    print('Total Parameters: {0:,}'.format(sum(conv_total_params) + sum(fc_total_params)))

    print('---------------- Done Pruning! ----------------\n')

    return model


def svd_weights(model, sv_truncation=0.25):

    print('---------------- Start SVD Compression! (SV Truncation = {0}%) ----------------'
          .format(round(sv_truncation * 100)))

    total_weights = 0
    svd_weights = 0

    # Sum the total weights
    for i in range(0, len(model.layers)):
        if "Conv" not in type(model.layers[i]).__name__ and "Dense" not in type(model.layers[i]).__name__:
            continue
        weights = model.layers[i].get_weights()

        for j in range(0, len(weights)):
            total_weights += weights[j].size

    for i in range(0, len(model.layers)):  # TODO change to zero
        if "Conv" not in type(model.layers[i]).__name__ and "Dense" not in type(model.layers[i]).__name__:
            continue

        weights = model.layers[i].get_weights()
        layer_weights_num = 0

        for j in range(0, len(weights)):
            layer_weights_num += weights[j].size

        for j in range(0, len(weights)):
            first_dim = np.power(2, np.floor(np.log2(np.sqrt(weights[j].size))))

            while (weights[j].size/first_dim).is_integer() == False and first_dim > 4:
                first_dim /= 2

            if first_dim == 4 and np.power(2, np.floor(np.log2(np.sqrt(weights[j].size)))) != 4:
                continue

            first_dim = int(min(first_dim, weights[j].size/first_dim))

            weights2d = weights[j].reshape([first_dim, -1])

            u, s, vh = np.linalg.svd(weights2d, full_matrices=True)

            s_cumsum = np.cumsum(np.flip(s))

            s_sum = np.sum(s)

            k = len(s) - np.argmax(s_cumsum/s_sum > sv_truncation)

            if k*(first_dim + weights2d.size/first_dim + 1) > weights2d.size:
                continue

            svd_weights += weights2d.size - k*(first_dim + weights2d.size/first_dim + 1)

            smat = np.zeros((first_dim, round(weights[j].size/first_dim)), s.dtype)

            smat[:k, :k] = np.diag(s[:k])

            weights2d_truncated = np.dot(u, np.dot(smat, vh))

            # approx = np.allclose(weights2d_truncated, weights2d)

            # print('DATA LOSS RATIO: {0}'.format(1-np.sum(s[0:round(len(s)*(1-compression_ratio))])/np.sum(s)))

            weights[j] = weights2d_truncated.reshape(weights[j].shape)

        model.layers[i].set_weights(weights)

    print('---------------- Done SVD Compression! SVD weights percent = {0}% ----------------\n'
          .format(round(svd_weights/total_weights*100)))

    return model


def quantize_weights(model, q=16, per_channel=False, symmetric_quantization=True):

    # Quantize the weights

    print(('Start Quantization Per Channel' if per_channel else 'Start Quantization') + ' to {0} bits!'.format(q))

    for i in range(0, len(model.layers)):
        if "Conv" not in type(model.layers[i]).__name__ and "Dense" not in type(model.layers[i]).__name__:
            continue

        weights = model.layers[i].get_weights()

        for j in range(0, len(weights)):

            if len(weights[j].shape) == 1:  # Skip biases quantization
                continue

            if "Conv" in type(model.layers[i]).__name__ and per_channel and weights[j].shape[0]*weights[j].shape[1] > 1:
                for cin in range(0, weights[j].shape[2]):
                    max_val = max(abs(weights[j][:, :, cin, :].min()), abs(weights[j][:, :, cin, :].max())) if \
                        symmetric_quantization else weights[j][:, :, cin, :].max()
                    min_val = -max_val if symmetric_quantization else weights[j][:, :, cin, :].min()

                    if max_val - min_val == 0:
                        continue

                    weights_quantize_step = (max_val - min_val) / 2 ** q

                    layer_weights_uint = np.round((weights[j][:, :, cin, :] - min_val) / (max_val - min_val) *
                                                  (2 ** q - 1))

                    layer_weights_quantized = layer_weights_uint * weights_quantize_step + min_val

                    weights[j][:, :, cin, :] = layer_weights_quantized
            else:
                max_val = max(abs(weights[j].min()), abs(weights[j].max())) if symmetric_quantization \
                                                                            else weights[j].max()
                min_val = -max_val if symmetric_quantization else weights[j].min()

                if max_val - min_val == 0:
                    continue

                weights_quantize_step = (max_val - min_val) / 2 ** q

                layer_weights_uint = np.round((weights[j] - min_val) / (max_val - min_val) * (2 ** q - 1))

                layer_weights_quantized = layer_weights_uint * weights_quantize_step + min_val

                weights[j] = layer_weights_quantized

        model.layers[i].set_weights(weights)

    print('Done Quantization!\n')

    return model


def save_weights_to_dat_file(model, architecture="", q=16, per_channel=False, symmetric_quantization=True):

    # Get all the weights from model

    weights_to_store = []

    for i in range(0, len(model.layers)):
        if "Conv" not in type(model.layers[i]).__name__ and "Dense" not in type(model.layers[i]).__name__:
            continue

        weights = model.layers[i].get_weights()

        for j in range(0, len(weights)):

            # Skip biases quantization
            if q == 32 or len(weights[j].shape) == 1:
                weights_to_store.append(weights[j])
                continue

            if "Conv" in type(model.layers[i]).__name__ and per_channel and weights[j].shape[0]*weights[j].shape[1] > 1:
                for cin in range(0, weights[j].shape[2]):
                    max_val = max(abs(weights[j][:, :, cin, :].min()), abs(weights[j][:, :, cin, :].max())) if \
                        symmetric_quantization else weights[j][:, :, cin, :].max()
                    min_val = -max_val if symmetric_quantization else weights[j][:, :, cin, :].min()

                    layer_quantize_weights_2sc = np.round((weights[j][:, :, cin, :] - min_val) / (max_val - min_val) *
                                                          (2 ** q - 1)) - 2 ** (q - 1)
                    if q == 8:
                        weights_to_store.append(layer_quantize_weights_2sc.astype(np.int8))
                    elif q == 16:
                        weights_to_store.append(layer_quantize_weights_2sc.astype(np.int16))
                    else:
                        weights_to_store.append(layer_quantize_weights_2sc.astype(np.int32))
            else:
                max_val = max(abs(weights[j].min()), abs(weights[j].max())) if symmetric_quantization \
                                                                            else weights[j].max()
                min_val = -max_val if symmetric_quantization else weights[j].min()

                layer_quantize_weights_2sc = np.round((weights[j] - min_val) / (max_val - min_val) *
                                                      (2 ** q - 1)) - 2 ** (q - 1)
                if q == 8:
                    weights_to_store.append(layer_quantize_weights_2sc.astype(np.int8))
                elif q == 16:
                    weights_to_store.append(layer_quantize_weights_2sc.astype(np.int16))
                else:
                    weights_to_store.append(layer_quantize_weights_2sc.astype(np.int32))

    # Save weights to dat file
    weight_file = open(r"weights\{0}weight_file.dat".format(architecture + '_' if architecture != "" else ""), "wb")
    for j in range(len(weights_to_store)):
        weight_file.write(weights_to_store[j].tobytes())

    print('Weights dat file was created!\n')


def plot_weights_histogram(model):

    weights = []

    for i in range(0, len(model.layers)):
        if "Conv" not in type(model.layers[i]).__name__ and "Dense" not in type(model.layers[i]).__name__:
            continue

        layer_weights = model.layers[i].get_weights()

        for w in layer_weights:
            weights = np.append(weights, w.flatten())

    weights_range = np.max(np.abs(weights))
    print('Mean: {0}'.format(np.mean(weights)))
    print('Std: {0}'.format(np.std(weights)))
    print('X Cover: {0}'.format(np.std(weights)/(weights_range-np.mean(weights))))
    # print('Cover: {0}'.format(np.std(weights)))
    plt.hist(weights, bins=100, range=[-weights_range, weights_range])
    plt.xlabel("Weight value")
    plt.ylabel("Weight appearances")
    plt.title("AlexNet weights histogram")
    plt.show()


def lra_per_layer(model:Model, layer_index=0, algorithm='tsvd') -> Model:
    assert layer_index < len(model.layers), 'ERROR lra_per_layer: given layer index is out of bounds'
    # print('---------------- Start Compression with {0} for layer {1}!) ----------------'.format(algorithm, layer_index))

    # if "Conv" not in type(model.layers[layer_index]).__name__ and \
    #         "Dense" not in type(model.layers[layer_index]).__name__:
    #     print('The selected layer is not Conv2D or FC layer')
    #     exit(-1)

    weights = model.layers[layer_index].get_weights()

    for j in range(0, len(weights)):

        if len(weights[j].shape) > 2:
            weights2d = weights[j].reshape([weights[j].shape[-1], -1])
        elif len(weights[j].shape) == 1:
            continue
        else:
            weights2d = weights[j]

        u, s, vh = np.linalg.svd(weights2d, full_matrices=True)

        k = sum(s > 0.001) - 1

        smat = np.zeros((u.shape[-1], vh.shape[0]), s.dtype)
        smat[:k, :k] = np.diag(s[:k])

        weights2d_truncated = np.dot(u, np.dot(smat, vh))  # TODO doesn't make sense, where do we save parameters?


        if len(weights[j].shape) > 2:
            weights[j] = weights2d_truncated.reshape(weights[j].shape)
        else:
            weights[j] = weights2d_truncated

    model.layers[layer_index].set_weights(weights)

    # print('---------------- Done Compression with {0} for layer {1}!) ----------------'.format(algorithm, layer_index))

    return model, k, len(s)


