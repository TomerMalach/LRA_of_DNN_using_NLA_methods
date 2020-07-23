from tensorflow.keras.models import Model
import numpy as np
import sys
from matplotlib import pyplot as plt
from logger_utils import get_logger
from scipy.linalg import qr, orth
from functools import reduce

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



lra_memory = {} # saves the decompositions
curr_num_of_params = {}  #  saves the num of params
weights_memory = {}


def svd(model: Model, layer_index=0, update_memory=False, martinson=None):
    weights = model.layers[layer_index].get_weights()

    total_params_per_layer = 0
    k = 0
    s = 0
    for j in range(0, len(weights)):

        if len(weights[j].shape) == 1:
            total_params_per_layer += weights[j].shape[0]
            continue

        # initialize the memory with recurring data
        if layer_index not in lra_memory.keys():
            if len(weights[j].shape) > 2:
                weights2d = np.zeros_like(weights[j].reshape([weights[j].shape[-1], -1]))
                for kernel in range(0, weights[j].shape[-1]):
                    weights2d[kernel, :] = weights[j][:, :, :, kernel].reshape([1, -1])
            else:
                weights2d = weights[j]

            # initial num of params
            (m, n) = np.shape(weights2d)
            curr_num_of_params[(layer_index, j)] = n * m

            # martinson randomization
            if martinson:
                for l in range(1, min(weights2d.shape))[::-1]:
                    ohm = np.random.randn(weights2d.shape[1], l)
                    projW = np.matmul(weights2d, ohm)
                    Q = orth(projW)
                    B = np.matmul(Q.T, weights2d)
                    weights_memory[(layer_index, j)] = B
                    # curr_num_of_params[(layer_index, j)]
            else:
                weights_memory[(layer_index, j)] = weights2d

            # calculating the svd
            u, s, vh = np.linalg.svd(weights2d, full_matrices=True)

            if martinson:
                u = np.matmul(Q, u)

            lra_memory[(layer_index, j)] = (u, s, vh)

        else:  # loading recurring data
            u, s, vh = lra_memory[(layer_index, j)]

        # truncation process
        k = sum(s > 0.001)
        (m, n) = np.shape(weights_memory[(layer_index, j)])
        num_of_params = (m + n) * k
        k -= np.ceil(len(s) * 0.005)
        k = np.clip(int(k), a_min=0, a_max=len(s))

        # only when when the change is real and not test
        if update_memory:
            s_update = s.copy()
            s_update[k:] = 0
            lra_memory[(layer_index, j)] = (u, s_update, vh)
            if num_of_params < curr_num_of_params[(layer_index, j)]:
                curr_num_of_params[(layer_index, j)] = num_of_params

        #  inverse transformation
        smat = np.zeros((u.shape[-1], vh.shape[0]), s.dtype)
        smat[:k, :k] = np.diag(s[:k])
        weights2d_truncated = np.dot(u, np.dot(smat, vh))
        if len(weights[j].shape) > 2:
            for kernel in range(0, weights[j].shape[-1]):
                weights[j][:, :, :, kernel] = \
                    weights2d_truncated[kernel, :].reshape(np.shape(weights[j][:, :, :, kernel]))
        else:
            weights[j] = weights2d_truncated


        total_params_per_layer += curr_num_of_params[(layer_index, j)]

    # update the true model if update memory if not update the temp model for testing\evaluation
    model.layers[layer_index].set_weights(weights)

    return model, k, len(s), total_params_per_layer


def create_inverse_permutation_matrix(p):
    permutation_mat = []
    for n in p:
        one_hot_vec = np.zeros(len(p))
        one_hot_vec[n] = 1
        permutation_mat.append(one_hot_vec)
    permutation_mat = np.asarray(permutation_mat)
    return permutation_mat

def rrqr(model, layer_index, update_memory):
    weights = model.layers[layer_index].get_weights()
    num_of_params = 0
    for j in range(0, len(weights)):

        if len(weights[j].shape) > 2:
            weights2d = np.zeros_like(weights[j].reshape([weights[j].shape[-1], -1]))
            for kernel in range(0, weights[j].shape[-1]):
                weights2d[kernel, :] = weights[j][:, :, :, kernel].reshape([1, -1])
            # weights2d = weights[j].reshape([weights[j].shape[-1], -1])
        elif len(weights[j].shape) == 1:
            num_of_params += weights[j].shape[0]
            continue
        else:
            weights2d = weights[j]

        if layer_index not in lra_memory.keys():
            q, r, p = qr(weights2d, pivoting=True)
            lra_memory[layer_index] = (q, r, p)
            tmp = 0
            for weight in weights:
                tmp += reduce(lambda x1, x2: x1 * x2, np.shape(weight))
            curr_num_of_params[layer_index] = tmp
        else:
            q, r, p = lra_memory[layer_index]

        k = sum(np.sum(np.abs(r), axis=1) > 0.001)
        k -= np.floor(np.shape(r)[1]) * 0.01
        k = np.clip(int(k), a_min=0, a_max=np.shape(r)[1])
        q_updated = q.copy()
        r_updated = r.copy()
        q_updated[:, k:] = 0  # eliminate silent columns
        r_updated[k:, :] = 0  # eliminate rows
        n = np.shape(r)[1]
        num_of_params = np.shape(q)[0] * k + ((k + 1) / 2 + (n - k)) * k  # q r num of params
        if update_memory:
            lra_memory[layer_index] = (q_updated, r_updated, p)
            if num_of_params < curr_num_of_params[layer_index]:
                curr_num_of_params[layer_index] = num_of_params

        permutation_inv = create_inverse_permutation_matrix(p)

        weights2d_truncated = np.matmul(np.matmul(q_updated, r_updated), permutation_inv)
        if len(weights[j].shape) > 2:
            for kernel in range(0, weights[j].shape[-1]):
                weights[j][:, :, :, kernel] = weights2d_truncated[kernel, :].reshape(np.shape(weights[j][:, :, :, kernel]))
            # weights[j] = weights2d_truncated.reshape(weights[j].shape)
        else:
            weights[j] = weights2d_truncated


    model.layers[layer_index].set_weights(weights)

    return model, k, np.shape(r)[1], curr_num_of_params[layer_index]


def lra_per_layer(model:Model, layer_index=0, algorithm='tsvd', update_memory=False):
    assert layer_index < len(model.layers), 'ERROR lra_per_layer: given layer index is out of bounds'
    if 'tsvd' in algorithm:
        return svd(model, layer_index, update_memory)
    elif 'rrqr' in algorithm:
        return rrqr(model, layer_index, update_memory)

