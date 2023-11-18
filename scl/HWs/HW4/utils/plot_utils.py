import matplotlib.pyplot as plt
import itertools

# import json
# import pandas as pd
# import seaborn as sns

color_methods = {
    'rho0.9_VQ': [1, 0, 0, 1],
    'rho0.99_VQ': [0, 1, 0, 1],
    'rho0.5_VQ': [0, 0, 1, 1],
    'rho0.9_RVQ': [1, 0, 0, 1],
    'rho0.99_RVQ': [0, 1, 0, 1],
    'rho0.5_RVQ': [0, 0, 1, 1],
    'TC_VQ': [0, 1, 0, 0.5]
}

# marker_block_shapes = {
#     1: 'o',
#     2: 's',
#     4: 'x',
#     5: 'd',
#     8: '^',
#     16: '<'
# }
marker_style_rho = [',', '+', '>', '<', '^', 'v', 'o', '*', 'x', 'd']

line_style_rho = {
    0.5: '.-',
    0.9: '--',
    0.99: ':'
}


def plot_results(results, expt_params, data_params, rate=1):
    rho_set = data_params['rho']

    VQ_expt_params = expt_params['VQ']
    RVQ_expt_params = expt_params['RVQ']

    if VQ_expt_params:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        plot_VQ_expts(fig, ax, results, VQ_expt_params, rho_set)
        plt.show()
    if RVQ_expt_params:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        # fig, ax = plot_VQ_expts(fig, ax, results, VQ_expt_params, rho_set, show_legend=False)
        plot_RVQ_expts(fig, ax, results, RVQ_expt_params, rho_set)
        # ax[0].grid()
        # ax[1].grid()
        plt.show()


def plot_VQ_expts(fig, ax, results, VQ_expt_params, rho_set, show_legend=True):
    for rho in rho_set:
        num_points = len(VQ_expt_params)
        marker_style = itertools.cycle(marker_style_rho[:num_points])

        block_sizes = []
        distortion_data, enc_time_data = [], []
        for block_size, _ in VQ_expt_params:
            block_sizes.append(block_size)
            distortion_data.append(results['mse_distortion'][f'rho{rho}']['VQ'][f'bs:{block_size}'])
            enc_time_data.append(results['encoding_time'][f'rho{rho}']['VQ'][f'bs:{block_size}'])

        marker = next(marker_style)
        # assert that we are plotting at fixed-rate
        # print(results['rate'][f'rho{rho}']['VQ'][f'bs:{block_size}'])
        assert results['rate'][f'rho{rho}']['VQ'][f'bs:{block_size}'][0] == 1.0

        ax[0].plot(block_sizes,
                   distortion_data,
                   color=color_methods[f'rho{rho}_VQ'],
                   marker='o',  # marker=marker_block_shapes[block_size],
                   linestyle='--',
                   label=f"Rho: {rho}")  # ,\nBlock Size={block_size}")

        ax[1].plot(enc_time_data[1:],
                   distortion_data[1:],
                   color=color_methods[f'rho{rho}_VQ'],
                   marker='o',  # marker=marker_block_shapes[block_size],
                   linestyle='--',
                   label=f"Rho: {rho}")  # ,\nBlock Size={block_size}")

    ax[0].set_xlabel('Block Size', fontsize=14)
    ax[0].set_ylabel('MSE Distortion', fontsize=14)
    # ax[0].set_title('VQ Performance (rate = 1)', fontsize=18)
    # ax[0].set_ylim([0, 0.4])

    ax[1].set_xlabel('Encoding Time (sec)', fontsize=14)
    ax[1].set_ylabel('MSE Distortion', fontsize=14)
    # ax[1].set_title('VQ Performance (rate = 1)', fontsize=18)
    # ax[1].set_xlim(0.4, 10)

    plt.suptitle('VQ Performance (rate = 1)', fontsize=18)

    if show_legend:
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=10)
        fig.subplots_adjust(right=0.9)

    ax[0].grid()
    ax[1].grid()

    ax[1].set_xscale('log')

    return fig, ax


def plot_RVQ_expts(fig, ax, results, RVQ_expt_params, rho_set):
    for rho in rho_set:
        # if rho == 0.99:
        #     continue

        num_points = len(RVQ_expt_params)
        marker_style = itertools.cycle(marker_style_rho[:num_points])

        for block_size, num_res_steps, _ in RVQ_expt_params:
            marker = next(marker_style)
            # assert that we are plotting at fixed-rate
            # print(results['rate'][f'rho{rho}']['RVQ'][f'bs:{block_size}_numsteps:{num_res_steps}'])
            assert results['rate'][f'rho{rho}']['RVQ'][f'bs:{block_size}_numsteps:{num_res_steps}'][0] == 1.0

            ax[0].scatter(block_size,
                          results['mse_distortion'][f'rho{rho}']['RVQ'][f'bs:{block_size}_numsteps:{num_res_steps}'],
                          color=color_methods[f'rho{rho}_RVQ'],
                          marker=marker,  # marker=marker_block_shapes[num_res_steps],
                          # linestyle='--',
                          label=f"Rho: {rho},\nBlockSize={block_size}, #ResSteps={num_res_steps}")
            ax[1].scatter(results['encoding_time'][f'rho{rho}']['RVQ'][f'bs:{block_size}_numsteps:{num_res_steps}'],
                          results['mse_distortion'][f'rho{rho}']['RVQ'][f'bs:{block_size}_numsteps:{num_res_steps}'],
                          color=color_methods[f'rho{rho}_RVQ'],
                          marker=marker,  # marker=marker_block_shapes[num_res_steps],
                          # linestyle='--',
                          label=f"Rho: {rho},\nBlockSize={block_size}, #ResSteps={num_res_steps}")

    ax[0].set_xlabel('Block Size', fontsize=14)
    ax[0].set_ylabel('MSE Distortion', fontsize=14)
    # ax[0].set_title('RVQ Performance (rate = 1)', fontsize=18)
    # ax[0].set_ylim([0, 0.4])

    ax[1].set_xlabel('Encoding Time (sec)', fontsize=14)
    ax[1].set_ylabel('MSE Distortion', fontsize=14)
    # ax[0].set_title('RVQ Performance (rate = 1)', fontsize=18)
    # ax[1].set_xlim(0.4, 10)

    plt.suptitle('RVQ Performance (rate = 1)', fontsize=18)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=6)
    fig.subplots_adjust(right=0.85)

    ax[0].grid()
    ax[1].grid()

    ax[1].set_xscale('log')

    return fig, ax

# def plot_results(results_path, **kwargs):
#     results = json.load(open(results_path, 'r'))
#     marker_block_shapes = {
#         1: 'o',
#         2: 's',
#         4: 'x',
#         5: 'd',
#         8: '^',
#         16: '<'
#     }
#     color_methods = {
#         'VQ': [1, 0, 0, 0.5],
#         'RVQ': [0, 0, 1, 0.5],
#         'TC_VQ': [0, 1, 0, 0.5]
#     }
#
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
#     fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
#     VQ_params = kwargs['VQ_params']
#     RVQ_params = kwargs['RVQ_params']
#     TC_VQ_params = kwargs['TC_VQ_params']
#
#
#     if VQ_params is not None:
#         for block_size, _ in VQ_params:
#             ax[0].scatter(results['rate']['VQ'][f'bs:{block_size}'],
#                           results['mse_distortion']['VQ'][f'bs:{block_size}'],
#                           color=color_methods['VQ'],  marker=marker_block_shapes[block_size],
#                           label=f"VQ: Block Size={block_size}")
#             ax[1].scatter(results['encoding_time']['VQ'][f'bs:{block_size}'],
#                           results['mse_distortion']['VQ'][f'bs:{block_size}'],
#                           color=color_methods['VQ'], marker=marker_block_shapes[block_size],
#                           label=f"VQ: Block Size={block_size}")
#             ax1[0].scatter(results['rate']['VQ'][f'bs:{block_size}'],
#                           results['mse_distortion']['VQ'][f'bs:{block_size}'],
#                           color=color_methods['VQ'], marker=marker_block_shapes[block_size],
#                           label=f"VQ: Block Size={block_size}")
#             ax1[1].scatter(results['encoding_time']['VQ'][f'bs:{block_size}'],
#                           results['mse_distortion']['VQ'][f'bs:{block_size}'],
#                           color=color_methods['VQ'], marker=marker_block_shapes[block_size],
#                           label=f"VQ: Block Size={block_size}")
#     if RVQ_params is not None:
#         for block_size, num_res_steps, _ in kwargs['RVQ_params']:
#             ax[0].scatter(results['rate']['RVQ'][f'bs:{block_size}_numsteps:{num_res_steps}'],
#                           results['mse_distortion']['RVQ'][f'bs:{block_size}_numsteps:{num_res_steps}'],
#                           color=color_methods['RVQ'], marker=marker_block_shapes[block_size],
#                           label=f"RVQ: Block Size={block_size},\nNum Res Steps={num_res_steps}")
#             ax[1].scatter(results['encoding_time']['RVQ'][f'bs:{block_size}_numsteps:{num_res_steps}'],
#                           results['mse_distortion']['RVQ'][f'bs:{block_size}_numsteps:{num_res_steps}'],
#                           color=color_methods['RVQ'], marker=marker_block_shapes[block_size],
#                           label=f"RVQ: Block Size={block_size},\nNum Res Steps={num_res_steps}")
#     if TC_VQ_params is not None:
#         for block_size, _, bitrate_split in TC_VQ_params:
#             ax1[0].scatter(results['rate']['TC_VQ'][f'bs:{block_size}_bitratesplit:{bitrate_split}'],
#                           results['mse_distortion']['TC_VQ'][f'bs:{block_size}_bitratesplit:{bitrate_split}'],
#                           color=color_methods['TC_VQ'], marker=marker_block_shapes[block_size],
#                           label=f"TC_VQ: Block Size={block_size}")
#             ax1[1].scatter(results['encoding_time']['TC_VQ'][f'bs:{block_size}_bitratesplit:{bitrate_split}'],
#                           results['mse_distortion']['TC_VQ'][f'bs:{block_size}_bitratesplit:{bitrate_split}'],
#                           color=color_methods['TC_VQ'], marker=marker_block_shapes[block_size],
#                           label=f"TC_VQ: Block Size={block_size},\nBitrate Split:{[bitrate_split[0]]}")
#
#     ax[0].set_xlabel('Rate', fontsize=14)
#     ax[0].set_ylabel('MSE Distortion', fontsize=14)
#     ax[0].set_title('Rate-Distortion (RD) Curve', fontsize=18)
#
#     ax[1].set_xlabel('Encoding Time (sec)', fontsize=14)
#     ax[1].set_ylabel('MSE Distortion', fontsize=14)
#     ax[1].set_title('RD Encoding Complexity', fontsize=18)
#
#     handles, labels = ax[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=6)
#     # ax[0].legend(fontsize=8)
#     # ax[1].legend(fontsize=8)
#     fig.subplots_adjust(right=0.85)
#     ax[0].grid()
#     ax[1].grid()
#
#     ax1[0].set_xlabel('Rate', fontsize=14)
#     ax1[0].set_ylabel('MSE Distortion', fontsize=14)
#     ax1[0].set_title('Rate-Distortion (RD) Curve', fontsize=18)
#
#     ax1[1].set_xlabel('Encoding Time (sec)', fontsize=14)
#     ax1[1].set_ylabel('MSE Distortion', fontsize=14)
#     ax1[1].set_title('RD Encoding Complexity', fontsize=18)
#
#     handles, labels = ax1[0].get_legend_handles_labels()
#     fig1.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=6)
#     # ax1[0].legend(fontsize=8)
#     # ax1[1].legend(fontsize=8)
#     fig1.subplots_adjust(right=0.85)
#     ax1[0].grid()
#     ax1[1].grid()
#
#     plt.show()
