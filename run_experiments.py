import os


if __name__ == '__main__':
    os.system('python main.py --path {} --lra_algo {} --arch {} --dataset {}'.format(r'alexnet/alexnet_cifar10_original.h5', 'tsvd', 'alexnet','cifar10'))
    os.system('python main.py --path {} --lra_algo {} --arch {} --dataset {}'.format(r'alexnet/alexnet_cifar10_strip_pruned_60.h5', 'tsvd', 'alexnet','cifar10'))
    os.system('python main.py --path {} --lra_algo {} --arch {} --dataset {}'.format(r'alexnet/alexnet_cifar10_strip_pruned_90.h5', 'tsvd', 'alexnet','cifar10'))