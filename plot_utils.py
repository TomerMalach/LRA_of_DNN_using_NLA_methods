from matplotlib import pyplot as plt
import numpy as np


def plot_score_versus_compression(save_dir, score_data, compression_data):
    fig = plt.figure()
    plt.plot(compression_data, score_data, marker='o', color='b')
    plt.xlabel('compression')
    plt.ylabel('score')
    plt.title('score as a function of SVD-compression')
    plt.grid()
    plt.show()
    fig.savefig('{0}/SVD_compression_vs_score_{0}_scaled.png'.format(save_dir))
    fig = plt.figure()
    plt.plot(compression_data, score_data, marker='o', color='b')
    plt.xlabel('compression')
    plt.ylabel('score')
    plt.title('score as a function of SVD-compression')
    plt.ylim(0,1)
    plt.grid()
    plt.show()
    fig.savefig('{0}/SVD_compression_vs_score_{0}.png'.format(save_dir))


# def plot_compression_vs_score_and_vs_condition_number_increase(save_dir, score_data,
#                                                                compression_data,
#                                                                condition_number_diff_data ):
#     fig, ax = plt.subplots()
#     color = 'tab:red'
#     ax.set_xlabel('compression ratio')
#     ax.set_ylabel('score')
#     ax.plot(compression_data, score_data, color=color)
#     ax.tick_params(axis='y', labelcolor=color)
#     ax.legend('score')
#     ax2 = ax.twinx()
#     color = 'tab:blue'
#     ax2.set_ylabel('condition number')
#     ax2.plot(compression_data, condition_number_diff_data, color=color)
#     ax2.set_yscale('log')
#     ax2.tick_params(axis='y', labelcolor=color)
#     ax2.legend('sum condition number')
#     fig.tight_layout()
#     plt.grid()
#     plt.show()
#     fig.savefig('{0}/SVD_compression_vs_score_and_vs_condition_number_{0}.png'.format(save_dir))