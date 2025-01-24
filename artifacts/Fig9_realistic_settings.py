import os

os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import argparse
import warnings

import pandas as pd
from math import log2, ceil

from tqdm import tqdm

from config import default_para as config
from matplotlib import pyplot as plt, rcParams, gridspec
from artifacts.Fig5_EvictRecord_prfm import runVORAM, runORAM


def prepare_csv(block_size_list, data_volume):
    df = pd.DataFrame(
        columns=['block_size', 'path_time', 'path_comm', 'ring_time', 'ring_comm', 'concur_time', 'concur_comm'])
    height_list = [ceil(log2(data_volume / block_size / config['bucket_size'])) for block_size in block_size_list]

    path_time = []
    path_comm = []
    ring_time = []
    ring_comm = []
    concur_time = []
    concur_comm = []

    with tqdm(total=sum(height_list) * config['test_interval'] * 5, ncols=80) as pbar:
        config['pbar'] = pbar
        for height, block_size in zip(height_list, block_size_list):
            size_str = f"{block_size / 1024}KB" if block_size >= 1024 else f"{block_size}B"
            pbar.set_description(f"Initializing {size_str} block")
            config['block_size'] = block_size
            results = runVORAM(height, **config)
            ring_time.append(results['ring_amortized'][0])
            concur_time.append(results['concur_amortized'][0])

            ring_comm.append(results['ring_amortized'][1])
            concur_comm.append(results['concur_amortized'][1])

            pbar.set_description(f"Height:\t{height}, Path ORAM\t\t")
            result = runORAM('path', height, **config)
            path_time.append(result[0])
            path_comm.append(result[1] + result[2])

    df['block_size'] = block_size_list
    df['path_time'] = path_time
    df['path_comm'] = path_comm
    df['ring_time'] = ring_time
    df['ring_comm'] = ring_comm
    df['concur_time'] = concur_time
    df['concur_comm'] = concur_comm

    df.to_csv(config['output_dir'] + 'various_block_size.csv', index=False)


def draw_figs(fig, axs, csv_path, rtt_list, ignore_legend=False, saveFig=True):
    df = pd.read_csv(config['output_dir'] + 'evict_record_time_comm.csv')

    path_time = list(df['path_time'])[-2]
    path_rtt = list(df['path_rtt'])[-2]
    ring_time = list(df['ring_amortized_time'])[-2]
    ring_rtt = list(df['ring_amortized_rtt'])[-2]
    concur_time = list(df['concur_amortized_time'])[-2]
    concur_rtt = list(df['concur_amortized_rtt'])[-2]

    for ax in axs:
        ax.tick_params(axis='x', which='major', pad=5)
        ax.grid(axis='y', color='#e6e6e6', linewidth=2, )
        ax.grid(axis='x', color='#e6e6e6', linewidth=2, )
        ax.set_xticks(rtt_list)
    axs[0].set_ylim(-210 / 30, 210)
    axs[0].set_yticks([0, 50, 100, 150, 200])
    axs[1].set_yscale('log', base=10)
    axs[1].set_yticks([0, 1, 10, 1e2, 1e3, 1e4])
    axs[1].set_ylim(2e-1, 5e4)
    axs[0].set_xlabel('Round-trip time (ms)')
    axs[1].set_xlabel('Block size (bytes)')

    path_real_time = [path_time * 1000 + path_rtt * rtt for rtt in rtt_list]
    ring_real_time = [ring_time * 1000 + ring_rtt * rtt for rtt in rtt_list]
    concur_real_time = [concur_time * 1000 + concur_rtt * rtt for rtt in rtt_list]

    df = pd.read_csv(csv_path)

    block_size_list = list(df['block_size'])
    path_time = [x * 1000 for x in list(df['path_time'])]
    ring_time = [x * 1000 for x in list(df['ring_time'])]
    concur_time = [x * 1000 for x in list(df['concur_time'])]

    axs[0].plot(rtt_list, path_real_time, ':s', linewidth=3, color='#91caff', markersize=10, label='Path')
    axs[0].plot(rtt_list, concur_real_time, '--^', linewidth=3, color='#0958d9', markersize=10, label='Concur')
    axs[0].plot(rtt_list, ring_real_time, '-o', linewidth=3, color='#4096ff', markersize=10, label='Ring')
    axs[0].set_ylabel('Response time (ms)')
    axs[0].set_title(f'# $2^{{{tested_height}}}$ buckets, 4kB block', fontsize=config['fig_config']['font.size'])

    axs[1].plot(block_size_list, path_time, ':s', linewidth=3, color='#ff85c0', markersize=10, label='Path')
    axs[1].plot(block_size_list, concur_time, '--^', linewidth=3, color='#c41d7f', markersize=10,
                label='Concur')
    axs[1].plot(block_size_list, ring_time, '-o', linewidth=3, color='#eb2f96', markersize=10, label='Ring')
    axs[1].set_xscale('log', base=2)
    axs[1].set_xticks([2 ** 6, 2 ** 9, 2 ** 12, 2 ** 15, 2 ** 18])
    axs[1].set_ylabel('Processing time (ms)')
    axs[1].set_title(f'# 4GB logical data', fontsize=config['fig_config']['font.size'])

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines = [lines[i] for i in [0, 3, 2, 5, 1, 4]]
    labels = [labels[i] for i in [0, 3, 2, 5, 1, 4]]
    legend = fig.legend(lines, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1),
                        fontsize=config['fig_config']['font.size'] * 0.88, labelspacing=0.25, handletextpad=0.25,
                        columnspacing=0.6)
    legend.get_frame().set_facecolor('#f7f7f7')
    legend.get_frame().set_edgecolor('#f7f7f7')

    plt.subplots_adjust(wspace=-.15)
    plt.tight_layout(pad=0.3)
    if not ignore_legend:
        plt.subplots_adjust(top=0.75)
    else:
        plt.subplots_adjust(top=0.77, bottom=0.3)
    if saveFig:
        fig.savefig(config['output_dir'] + 'Fig9_realistic_settings.pdf')


def main():
    parser = argparse.ArgumentParser(description='Prepare the data used by Figure-11 and draw the resultant figures.')
    parser.add_argument('-p', '--prepare', action='store_true', help='Prepare the data.')
    parser.add_argument('-d', '--draw', action='store_true', help='Draw the resultant figures.')
    parser.add_argument('-c', '--compare', action='store_true',
                        help='Generate comparison with the graphs in the paper.')
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

    if args.prepare:
        print("# Preparing the data ... (Initialization could take a while)")
        prepare_csv(block_size_list, data_volume)
        print(f"\tData saved to {config['output_dir'] + 'various_block_size.csv'}\n")

    if args.draw:
        print("# Drawing the figures ...")
        rcParams.update(config['fig_config'])
        fig, axs = plt.subplots(ncols=2, figsize=(8.4, 4.34))
        draw_figs(fig, axs, config['output_dir'] + 'various_block_size.csv', rtt_list)
        print(f"\tFigure saved to {config['output_dir'] + 'Fig9_realistic_settings.pdf'}\n")

        if not args.compare:
            plt.show()

    if args.compare:
        print("# Comparing the results with the graphs in the paper ...")
        rcParams.update(config['fig_config'])

        mid = 0.1
        fig = plt.figure(figsize=(8.4 * 2 + mid, 4.8))
        gs = gridspec.GridSpec(1, 5, figure=fig, width_ratios=[4.2, 4.2, mid, 4.2, 4.2])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax_mid = fig.add_subplot(gs[0, 2])
        ax_mid.axis('off')
        ax3 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[0, 4])

        axs = [ax1, ax2, ax3, ax4]
        draw_figs(fig, axs[:2], config['output_dir'] + 'various_block_size.csv', rtt_list, saveFig=False)
        draw_figs(fig, axs[2:], config['paper_data_dir'] + 'various_block_size.csv', rtt_list, ignore_legend=True,
                  saveFig=False)
        fig.text(0.25, 0.05, 'Artifact Evaluation', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.text(0.75, 0.05, 'Paper Figure 9', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.savefig(config['output_dir'] + 'Fig9_realistic_settings_compare.png')
        print(f"\tFigure saved to {config['output_dir'] + 'Fig9_realistic_settings_compare.png'}\n")
        print("\tThe left is the figures generated by the artifact.\n"
              "\tThe right is the figures in the paper.")
        plt.show()


if __name__ == '__main__':
    tested_height = 18
    data_volume = 2 ** tested_height * config['block_size'] * config['bucket_size']
    rtt_list = [0, 5, 10, 15, 20, 25, 30]  # in ms
    block_size_list = [2 ** x for x in range(6, 19, 3)]
    block_size_list.reverse()
    config['test_interval'] = 2 ** 8  # For runtime saving
    main()
