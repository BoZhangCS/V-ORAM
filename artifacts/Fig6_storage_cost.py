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
from tqdm import tqdm
from math import log2
from os import urandom
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec
from random import random, choice, randint
from config import default_para as config
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from src.ConcurORAM import ConcurORAM
from src.Path_ORAM import Path_ORAM
from src.Ring_ORAM import Ring_ORAM


def prepare_csv(height_list):
    block_size = config.get('block_size')
    bucket_size = config.get('bucket_size')
    s_num = config.get('s_num')
    a_num = config.get('a_num')
    c_batch = config.get('c_batch')
    address_size = config.get('address_size')  # bytes
    test_interval = config.get('test_interval')
    path_stash = []
    ring_stash = []
    concur_stash = []

    with tqdm(total=sum(height_list) * 3 * test_interval, ncols=80) as pbar:
        for height in height_list:
            pbar.set_description(f"Height:\t{height}, Path ORAM\t")
            path_oram = Path_ORAM(height)
            real_datasets = {}
            maxStashSize = -1
            for i in range(test_interval):
                p_map = path_oram.position_map
                if random() < 0.5:
                    address = randint(0, len(p_map) - 1)
                    data = urandom(block_size)
                    path_oram.access('write', address, data)
                    real_datasets[address] = data
                else:
                    if len(real_datasets) == 0:
                        i -= 1
                        continue
                    address = choice(list(real_datasets.keys()))
                    path_oram.access('read', address, urandom(block_size))
                maxStashSize = max(maxStashSize, len(path_oram.stash))
                pbar.update(height)
            path_stash.append(maxStashSize * block_size)

        for height in height_list:
            pbar.set_description(f"Height:\t{height}, Ring ORAM\t")
            ring_oram = Ring_ORAM(height)
            real_datasets = {}
            maxStashSize = -1
            for i in range(test_interval):
                p_map = ring_oram.position_map
                if random() < 0.5:
                    address = randint(0, len(p_map) - 1)
                    data = urandom(block_size)
                    ring_oram.access('write', address, data)
                    real_datasets[address] = data
                else:
                    if len(real_datasets) == 0:
                        i -= 1
                        continue
                    address = choice(list(real_datasets.keys()))
                    ring_oram.access('read', address, urandom(block_size))
                maxStashSize = max(maxStashSize, len(ring_oram.stash))
                pbar.update(height)
            ring_stash.append(maxStashSize * block_size)

        for height in height_list:
            pbar.set_description(f"Height:\t{height}, ConcurORAM\t")
            concor = ConcurORAM(height=height, bucket_size=bucket_size, block_size=block_size,
                                c_batch=c_batch, a_num=a_num, s_num=s_num)
            p_map = concor.position_map

            real_datasets = {}
            maxStashSize = -1
            for i in range(test_interval // c_batch):
                batch_requests = []
                while len(batch_requests) < c_batch:
                    if random() < 0.5:
                        address = randint(0, len(p_map) - 1)
                        data = urandom(block_size)
                        real_datasets[address] = data
                        batch_requests.append(('write', address, data))
                    else:
                        if len(real_datasets) == 0:
                            continue
                        address = choice(list(real_datasets.keys()))
                        real = real_datasets[address]
                        batch_requests.append(('read', address, real))
                maxStashSize = max(maxStashSize, concor.union_size)
                pbar.update(height * c_batch)
            concur_stash.append(maxStashSize * block_size)

    position_map_size = [2 ** height * (address_size + height) for height in height_list]
    record_map_size = [2 ** height * int(log2(bucket_size)) for height in height_list]

    df = pd.DataFrame()
    df['height'] = [2 ** i for i in height_list]
    df['path_stash'] = path_stash
    df['ring_stash'] = ring_stash
    df['concur_stash'] = concur_stash
    df['pos_size'] = position_map_size
    df['rec_size'] = record_map_size
    df.to_csv(config['output_dir'] + 'storage_cost.csv', index=False)


def draw_figs(fig, axs, csv_path, ignore_legend=False, saveFig=True):
    df = pd.read_csv(csv_path)
    colors = ['#cce7af', '#88c5b2', '#08979c']
    # Common settings for all subplots
    for ax in axs:
        ax.tick_params(axis='x', which='major', pad=5)
        ax.grid(axis='y', linestyle='--', color='#e6e6e6', linewidth=2, zorder=-1)
        ax.grid(axis='x', linestyle='--', color='#e6e6e6', linewidth=2, zorder=-1)
        ax.set_xlabel('Buckets')
        ax.set_xscale('log', base=2)
        ax.set_ylim(-40 / 30, 40)
        ax.set_xticks([2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20])

    for ax in axs[1:]:
        ax.tick_params(labelleft=False)

    # Plot for Path ORAM
    bucket_number = df['height']
    path_stash = df['path_stash'] / 2 ** 20
    path_pos = df['pos_size'] / 2 ** 20
    path_rec = df['rec_size'] / 2 ** 20
    path_labels = ['Stash', 'Position map', 'Record map']
    axs[0].stackplot(bucket_number, path_stash, path_pos, path_rec, labels=path_labels, colors=colors)
    axs[0].set_ylabel('Storage cost (MB)')
    axs[0].set_title('Path ORAM', fontsize=config['fig_config']['font.size'])

    # Plot for Ring ORAM
    ring_stash = df['ring_stash'] / 2 ** 20
    ring_pos = df['pos_size'] / 2 ** 20
    ring_rec = df['rec_size'] / 2 ** 20
    axs[1].stackplot(bucket_number, ring_stash, ring_pos, ring_rec, colors=colors)
    axs[1].set_title('Ring ORAM', fontsize=config['fig_config']['font.size'])

    # Adding inset axis
    axins = inset_axes(axs[1], width="30%", height="40%", bbox_to_anchor=(1, 0.1, 1.1, 1),
                       bbox_transform=axs[0].transAxes, loc='center')
    axins.stackplot(bucket_number, ring_stash, ring_pos, ring_rec, colors=colors)
    axins.set_xlim(2 ** 14, 2 ** 16)
    axins.set_ylim(0, 0.1)
    axins.set_xscale('log', base=2)
    axins.set_xticks([2 ** 14, 2 ** 15, 2 ** 16])
    axins.xaxis.set_visible(False)
    mark_inset(axs[1], axins, loc1=3, loc2=4, fc="none", ec="0", zorder=10)

    # Plot for ConcurORAM
    concur_stash = df['concur_stash'] / 2 ** 20
    concur_pos = df['pos_size'] / 2 ** 20
    concur_rec = df['rec_size'] / 2 ** 20
    axs[2].stackplot(bucket_number, concur_stash, concur_pos, concur_rec, colors=colors)
    axs[2].set_title('ConcurORAM', fontsize=config['fig_config']['font.size'])

    lines_labels = [ax.get_legend_handles_labels() for ax in axs]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    if not ignore_legend:
        legend = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3)
        legend.get_frame().set_facecolor('#f7f7f7')
        legend.get_frame().set_edgecolor('#f7f7f7')

    plt.tight_layout(pad=0.2)
    if not ignore_legend:
        plt.subplots_adjust(top=0.78)
    else:
        plt.subplots_adjust(top=0.8, bottom=0.27, wspace=0.1)
    if saveFig:
        fig.savefig(config['output_dir'] + 'Fig6_storage_cost.pdf')


def main():
    parser = argparse.ArgumentParser(description='Prepare the data used by Figure-11 and draw the resultant figures.')
    parser.add_argument('-p', '--prepare', action='store_true', help='Prepare the data.')
    parser.add_argument('-d', '--draw', action='store_true', help='Draw the resultant figures.')
    parser.add_argument('-c', '--compare', action='store_true',
                        help='Generate comparison with the graphs in the paper.')
    parser.add_argument('-s', '--start', type=int, default=14, help='Start height (default: 14)')
    parser.add_argument('-e', '--end', type=int, default=20, help='End height (default: 20)')
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

    if args.prepare:
        print("# Preparing the data ...")
        start_height = args.start
        end_height = args.end
        prepare_csv(list(range(start_height, end_height + 1, 2)))
        print(f"\tData saved to {config['output_dir'] + 'storage_cost.csv'}\n")

    if args.draw:
        print("# Drawing the figures ...")
        rcParams.update(config['fig_config'])
        fig, axs = plt.subplots(1, 3, figsize=(7.9605, 4.7), sharey='all')
        draw_figs(fig, axs, config['output_dir'] + 'storage_cost.csv')
        print(f"\tFigure saved to {config['output_dir'] + 'Fig6_storage_cost.pdf'}\n")

        if not args.compare:
            plt.show()

    if args.compare:
        print("# Comparing the results with the graphs in the paper ...")
        rcParams.update(config['fig_config'])
        mid = 1
        fig = plt.figure(figsize=(18 + mid, 4.7 * 1.2))
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
        draw_figs(fig, axs[:3], config['output_dir'] + 'storage_cost.csv', saveFig=False)
        draw_figs(fig, axs[4:], config['paper_data_dir'] + 'storage_cost.csv',
                  ignore_legend=True, saveFig=False)
        fig.text(0.25, 0.05, 'Artifact Evaluation', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.text(0.75, 0.05, 'Paper Figure 6', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.savefig(config['output_dir'] + 'Fig6_storage_cost_compare.png')
        print(f"\tFigure saved to {config['output_dir'] + 'Fig6_storage_cost_compare.png'}\n")
        print("\tThe left is the figures generated by the artifact.\n"
              "\tThe right is the figures in the paper.")
        plt.show()


if __name__ == '__main__':
    main()
