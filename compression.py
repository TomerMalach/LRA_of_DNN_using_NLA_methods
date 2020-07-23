from tensorflow.keras.models import Model
import numpy as np
import sys
from matplotlib import pyplot as plt
from logger_utils import get_logger
from scipy.linalg import svd, qr, orth, norm
from functools import reduce
import time
from sklearn import preprocessing

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
shapes_memory = {}


def svd_per_layer(model: Model, layer_index=0, update_memory=False, martinson=False):
    weights = model.layers[layer_index].get_weights()

    total_params_per_layer = 0
    k = 0
    s = 0
    for j in range(0, len(weights)):

        if len(weights[j].shape) == 1:
            total_params_per_layer += weights[j].shape[0]
            continue

        # initialize the memory with recurring data
        if (layer_index, j) not in lra_memory.keys():
            if len(weights[j].shape) > 2:
                weights2d = np.zeros_like(weights[j].reshape([weights[j].shape[-1], -1]))
                for kernel in range(0, weights[j].shape[-1]):
                    weights2d[kernel, :] = weights[j][:, :, :, kernel].reshape([1, -1])
            else:
                weights2d = weights[j]

            # initial num of params
            (m, n) = np.shape(weights2d)
            curr_num_of_params[(layer_index, j)] = n * m
            shapes_memory[(layer_index, j)] = (m, n)

            # martinsson randomization
            if martinson or True: # or min(n, m) > 1000:
                # before = time.time()

                Q = np.zeros((m, n))
                B = np.zeros((n, n))
                initial_norm = norm(weights2d)
                rnd_mat = np.random.randn(n, m)
                rank = 0

                while norm(weights2d) > 0.01*initial_norm and rank < min(m, n):
                    # before = time.time()
                    w = rnd_mat[:, rank]
                    # print("\nrand time " + str(time.time() - before))

                    # before = time.time()
                    y = weights2d.dot(w)
                    # print("\nmatmul time " + str(time.time() - before))

                    # before = time.time()
                    q = y/norm(y)
                    # print("\nnorm time " + str(time.time() - before))

                    # before = time.time()
                    b = q.T.dot(weights2d)
                    # print("\nb time " + str(time.time() - before))

                    Q[:, rank] = q
                    B[rank, :] = b
                    # before = time.time()
                    # if Q == []:
                    #     Q = q
                    #     B = b
                    # else:
                    #     Q = np.concatenate((Q, q), axis=1)
                    #     B = np.concatenate((B, b), axis=0)
                    # print("\nallocate time " + str(time.time() - before))
                    weights2d = weights2d - np.matmul(np.expand_dims(q, 1), np.expand_dims(b, 0))
                    rank += 1

                # for l in range(1, min(weights2d.shape)+1)[::-1]:
                #     ohm = np.random.randn(n, l)
                #     projW = np.matmul(weights2d, ohm)
                #     Q_temp = orth(projW)
                #     B_temp = np.matmul(Q_temp.T, weights2d)
                #     if norm(weights2d-np.matmul(Q_temp, B_temp)) > 0.5:
                #         break
                #     # print('l={0}  -  norm={1}'.format(l, norm(weights2d-np.matmul(Q, B))))
                #     B = B_temp
                #     Q = Q_temp

                # calculating the svd
                u, s, vh = svd(B[0:rank, :], full_matrices=True)
                u = np.matmul(Q[:, 0:rank], u)
                # print("\nlayer {0} martinsson svd time ".format(model.layers[layer_index].name) + str(time.time() - before))

            else:

                # calculating the svd
                # before = time.time()
                u, s, vh = svd(weights2d, full_matrices=True)
                # print("\nlayer {0} svd time ".format(model.layers[layer_index].name) + str(time.time() - before))

            lra_memory[(layer_index, j)] = (u, s, vh)

        else:  # loading recurring data
            u, s, vh = lra_memory[(layer_index, j)]

        # truncation process
        k = sum(s > 0.001)
        (m, n) = shapes_memory[(layer_index, j)]
        num_of_params = (m + n) * k
        k -= np.ceil(len(s) * 0.1)
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


def rrqr_per_layer(model, layer_index, update_memory):
    weights = model.layers[layer_index].get_weights()
    total_params_per_layer = 0
    for j in range(0, len(weights)):

        if len(weights[j].shape) == 1:
            total_params_per_layer += weights[j].shape[0]
            continue

        if (layer_index, j) not in lra_memory.keys():
            if len(weights[j].shape) > 2:
                weights2d = np.zeros_like(weights[j].reshape([weights[j].shape[-1], -1]))
                for kernel in range(0, weights[j].shape[-1]):
                    weights2d[kernel, :] = weights[j][:, :, :, kernel].reshape([1, -1])
            else:
                weights2d = weights[j]

            # initial num of params
            (m, n) = np.shape(weights2d)
            curr_num_of_params[(layer_index, j)] = n * m
            # calculating the qr
            q, r, p = qr(weights2d, pivoting=True)
            lra_memory[(layer_index, j)] = (q, r, p)
        else: # loading recurring data
            q, r, p = lra_memory[(layer_index, j)]

        # truncation process
        k = sum(np.sum(np.abs(r), axis=1) > 0.001)
        k -= np.ceil(np.shape(r)[1] * 0.005)
        k = np.clip(int(k), a_min=0, a_max=np.shape(r)[1])
        q_updated = q.copy()
        r_updated = r.copy()
        q_updated[:, k:] = 0  # eliminate silent columns
        r_updated[k:, :] = 0  # eliminate rows
        n = np.shape(r)[1]
        num_of_params = np.shape(q)[0] * k + ((k + 1) / 2 + (n - k)) * k  # QR num of params

        # only when when the change is real and not test
        if update_memory:
            lra_memory[(layer_index, j)] = (q_updated, r_updated, p)
            if num_of_params < curr_num_of_params[(layer_index, j)]:
                curr_num_of_params[(layer_index, j)] = num_of_params

        #  inverse transformation
        permutation_inv = create_inverse_permutation_matrix(p)
        weights2d_truncated = np.matmul(np.matmul(q_updated, r_updated), permutation_inv)
        if len(weights[j].shape) > 2:
            for kernel in range(0, weights[j].shape[-1]):
                weights[j][:, :, :, kernel] = weights2d_truncated[kernel, :].reshape(np.shape(weights[j][:, :, :, kernel]))
        else:
            weights[j] = weights2d_truncated

        total_params_per_layer += curr_num_of_params[(layer_index, j)]


    model.layers[layer_index].set_weights(weights)

    return model, k, np.shape(r)[1], total_params_per_layer


def lra_per_layer(model:Model, layer_index=0, algorithm='tsvd', update_memory=False):
    assert layer_index < len(model.layers), 'ERROR lra_per_layer: given layer index is out of bounds'
    if 'tsvd' in algorithm:
        return svd_per_layer(model, layer_index, update_memory)
    elif 'rrqr' in algorithm:
        return rrqr_per_layer(model, layer_index, update_memory)

