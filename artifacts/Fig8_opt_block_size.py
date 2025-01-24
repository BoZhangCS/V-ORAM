import os

os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import argparse
import warnings
from math import log2, ceil
from config import default_para as config
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams, gridspec


def getBlowUp(file_size, block_size, bucket_size, a_num, s_num, c_batch, mas_stash_size):
    height = ceil(log2(total_volume // block_size // bucket_size * 2))
    path_path = height * block_size * bucket_size
    ring_bucket = concur_bucket = block_size * (bucket_size + s_num)
    ring_path = concur_path = height * ring_bucket

    path_cost = path_path * 2
    ring_cost = block_size + (1 / a_num + 1 / s_num) * ring_path * 2
    concur_cost = block_size * (
            5 + 6 * c_batch + (2 + 1 / c_batch) * (c_batch + mas_stash_size) + 2 * mas_stash_size) + 1 / c_batch * (
                          concur_path * 2 + concur_bucket * log2(c_batch) + block_size * 2)

    if file_size < block_size:
        path_blowup = path_cost / file_size
        ring_blowup = ring_cost / file_size
        concur_blowup = concur_cost / file_size
    else:
        path_blowup = path_cost / block_size
        ring_blowup = ring_cost / block_size
        concur_blowup = concur_cost / block_size

    return path_blowup, ring_blowup, concur_blowup


total_volume = 10 * 2 ** 30


def prepare_csv(block_size_list):
    df = pd.DataFrame()
    bucket_size = 4
    a_num = 3
    s_num = 6
    c_batch = 3
    maxStashSize = 32
    paths = []
    rings = []
    concurs = []
    for block_size in block_size_list:
        file_size = 64 * 2 ** 10
        block_size = 2 ** (block_size + 10)
        tmp = getBlowUp(file_size, block_size, bucket_size, a_num, s_num, c_batch, maxStashSize)
        paths.append(tmp[0])
        rings.append(tmp[1])
        concurs.append(tmp[2])
    df['b4_path'] = paths
    df['b4_ring'] = rings
    df['b4_concur'] = concurs

    bucket_size = 8
    a_num = 8
    s_num = 12
    c_batch = 8
    maxStashSize = 41
    paths = []
    rings = []
    concurs = []
    for block_size in block_size_list:
        file_size = 64 * 2 ** 10
        block_size = 2 ** (block_size + 10)
        tmp = getBlowUp(file_size, block_size, bucket_size, a_num, s_num, c_batch, maxStashSize)
        paths.append(tmp[0])
        rings.append(tmp[1])
        concurs.append(tmp[2])
    df['b8_path'] = paths
    df['b8_ring'] = rings
    df['b8_concur'] = concurs

    bucket_size = 16
    a_num = 20
    s_num = 28
    c_batch = 20
    maxStashSize = 65
    paths = []
    rings = []
    concurs = []
    for block_size in block_size_list:
        file_size = 64 * 2 ** 10
        block_size = 2 ** (block_size + 10)
        tmp = getBlowUp(file_size, block_size, bucket_size, a_num, s_num, c_batch, maxStashSize)
        paths.append(tmp[0])
        rings.append(tmp[1])
        concurs.append(tmp[2])
    df['b16_path'] = paths
    df['b16_ring'] = rings
    df['b16_concur'] = concurs

    df['block_size'] = [2 ** i for i in block_size_list]
    df.to_csv(config['output_dir'] + 'opt_block_size.csv', index=False)


def draw_figs(fig, axs, csv_path, ignore_legend=False, saveFig=True):
    df = pd.read_csv(csv_path)
    # Common settings for all subplots
    for ax in axs:
        ax.tick_params(axis='x', which='major', pad=5)
        ax.grid(axis='y', linestyle='--', color='#e6e6e6', linewidth=2, zorder=-1)
        ax.grid(axis='x', linestyle='--', color='#e6e6e6', linewidth=2, zorder=-1)
        ax.set_xlabel('Block size (KB)')
        # ax.set_ylim([0, 4500])
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=10)
        ax.set_xticks([2 ** 2, 2 ** 6, 2 ** 10])
        # ax.set_xticklabels([2 ** 2, 2 ** 6, 2 ** 10])

    axs[0].plot(df['block_size'], df['b4_concur'], ':>', linewidth=3, color='#91caff', markersize=10,
                label='Concur')
    axs[0].plot(df['block_size'], df['b4_ring'], '--s', linewidth=3, color='#4096ff', markersize=10,
                label='Ring')
    axs[0].plot(df['block_size'], df['b4_path'], '-o', linewidth=3, color='#0958d9', markersize=10,
                label='Path')
    axs[0].set_ylabel('Comm. blowup')
    axs[0].set_title('# $para~(4,3,6,3)$', fontsize=config['fig_config']['font.size'])

    axs[1].plot(df['block_size'][:-1], df['b8_concur'][:-1], ':>', linewidth=3, color='#91caff', markersize=10)
    axs[1].plot(df['block_size'], df['b8_ring'], '--s', linewidth=3, color='#4096ff', markersize=10)
    axs[1].plot(df['block_size'], df['b8_path'], '-o', linewidth=3, color='#0958d9', markersize=10)
    axs[1].set_title('# $para~(8,8,12,8)$', fontsize=config['fig_config']['font.size'])

    axs[2].plot(df['block_size'][:-1], df['b16_concur'][:-1], ':>', linewidth=3, color='#91caff', markersize=10)
    axs[2].plot(df['block_size'], df['b16_ring'], '--s', linewidth=3, color='#4096ff', markersize=10)
    axs[2].plot(df['block_size'][:-1], df['b16_path'][:-1], '-o', linewidth=3, color='#0958d9', markersize=10)
    axs[2].set_title('# $para~(16,20,28,20)$', fontsize=config['fig_config']['font.size'])

    lines_labels = [ax.get_legend_handles_labels() for ax in axs]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    legend = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3)
    legend.get_frame().set_facecolor('#f7f7f7')
    legend.get_frame().set_edgecolor('#f7f7f7')

    plt.tight_layout(pad=0.2)
    if not ignore_legend:
        plt.subplots_adjust(wspace=0.1, top=0.75)
    else:
        plt.subplots_adjust(wspace=0.1, top=0.77, bottom=0.33)
    if saveFig:
        fig.savefig(config['output_dir'] + 'Fig8_opt_block_size.pdf')


def main():
    parser = argparse.ArgumentParser(description='Prepare the data used by Figure-11 and draw the resultant figures.')
    parser.add_argument('-p', '--prepare', action='store_true', help='Prepare the data.')
    parser.add_argument('-d', '--draw', action='store_true', help='Draw the resultant figures.')
    parser.add_argument('-c', '--compare', action='store_true',
                        help='Generate comparison with the graphs in the paper.')
    parser.add_argument('-s', '--start', type=int, default=2, help='Start block size (default: 2^2)')
    parser.add_argument('-e', '--end', type=int, default=11, help='End block size (default: 2^11)')
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

    if args.prepare:
        print("# Preparing the data ...")
        start_block_size = args.start
        end_block_size = args.end
        block_size_list = list(range(start_block_size, end_block_size + 1))
        prepare_csv(block_size_list)
        print(f"\tData saved to {config['output_dir'] + 'opt_block_size.csv'}\n")

    if args.draw:
        print("# Drawing the figures ...")
        rcParams.update(config['fig_config'])
        fig, axs = plt.subplots(1, 3, figsize=(9, 4), sharey='all')
        draw_figs(fig, axs, config['output_dir'] + 'opt_block_size.csv')
        print(f"\tFigure saved to {config['output_dir'] + 'Fig8_opt_block_size.pdf'}\n")

        if not args.compare:
            plt.show()

    if args.compare:
        print("# Comparing the results with the graphs in the paper ...")
        rcParams.update(config['fig_config'])
        mid = 1.5
        fig = plt.figure(figsize=(18 + mid, 4.6))
        gs = gridspec.GridSpec(1, 7, figure=fig, width_ratios=[3, 3, 3, mid, 3, 3, 3])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
        ax_mid = fig.add_subplot(gs[0, 3])
        ax_mid.axis('off')
        ax4 = fig.add_subplot(gs[0, 4], sharey=ax1)
        ax5 = fig.add_subplot(gs[0, 5], sharey=ax1)
        ax6 = fig.add_subplot(gs[0, 6], sharey=ax1)

        axs = [ax1, ax2, ax3, ax_mid, ax4, ax5, ax6]
        draw_figs(fig, axs[:3], config['output_dir'] + 'opt_block_size.csv', saveFig=False)
        draw_figs(fig, axs[4:], config['paper_data_dir'] + 'opt_block_size.csv', ignore_legend=True, saveFig=False)
        fig.text(0.25, 0.05, 'Artifact Evaluation', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.text(0.75, 0.05, 'Paper Figure 8', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.savefig(config['output_dir'] + 'Fig8_opt_block_size_compare.png')
        print(f"\tFigure saved to {config['output_dir'] + 'Fig8_opt_block_size_compare.png'}\n")
        print("\tThe left is the figures generated by the artifact.\n"
              "\tThe right is the figures in the paper.")
        plt.show()


if __name__ == '__main__':
    main()
