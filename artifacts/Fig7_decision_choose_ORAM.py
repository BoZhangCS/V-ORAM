import os

os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import argparse
import warnings
from math import ceil

import pandas as pd
import time
from os import urandom

from pandas import DataFrame
from tqdm import tqdm

from config import default_para as config
from matplotlib import pyplot as plt, rcParams, gridspec
from src.utils import Encoder


def encryption_speed(block_num):
    data = urandom(4096)
    encoder = Encoder()
    for _ in range(block_num):
        encoder.Enc(data, 'path')


def decryption_speed(block_num):
    encoder = Encoder()
    cipher = encoder.Enc(urandom(4096), 'path')
    for _ in range(block_num):
        encoder.Dec(cipher)


def single_thread_encryption_speed():
    total_blocks = 2 ** 12
    start_time = time.time()
    encryption_speed(total_blocks)
    enc_speed = total_blocks * 4096 / (time.time() - start_time) * 1000
    start_time = time.time()
    decryption_speed(total_blocks)
    dec_speed = total_blocks * 4096 / (time.time() - start_time) * 1000
    return dec_speed, enc_speed


# The standard billing, data is billed by traffic
# Settings:
#   Comp: (fixed) AliCloud ecs.c7.2xlarge, 8vCPU, 16 GiB, 0.346032 $/hr; (flex) 2vCPU, 4 GiB, 0.086508 $/hr
#   Comm: (bandwidth) 5 Mbps, 0.0296 $/hr; (traffic) 0.153 $/GB
stand_traffic_price = 0.153
stand_bw = 10 * 2 ** 20


def get_bw(price):
    # return price / 0.0001805555556 * 2 ** 20
    if price < 0.0296:
        return (price / 0.00592) * 2 ** 20
    else:
        return ((price - 0.0296) / 0.021 + 5) * 2 ** 20


def get_comm_price(bw):
    # return 0.0001805555556 * bw / 2 ** 20
    if bw < 5 * 2 ** 20:
        return 0.00592 * (bw / 2 ** 20)
    else:
        return 0.0296 + 0.021 * (ceil(bw / 2 ** 20) - 5)


# Our strategy includes bandwidth billing
cost_saving = pd.read_csv(config['output_dir'] + 'evict_record_time_comm.csv')
path_down = list(cost_saving['path_down'])[-1]
path_up = list(cost_saving['path_up'])[-1]
path_rtt = list(cost_saving['path_rtt'])[-1]
ring_down = list(cost_saving['ring_down'])[-1]
ring_up = list(cost_saving['ring_up'])[-1]
ring_rtt = list(cost_saving['ring_rtt'])[-1]
concur_down_sync = list(cost_saving['concur_down_sync'])[-1]
concur_down_async = list(cost_saving['concur_down_async'])[-1]
concur_up_sync = list(cost_saving['concur_up_sync'])[-1]
concur_up_async = list(cost_saving['concur_up_async'])[-1]
concur_rtt_sync = list(cost_saving['concur_rtt_sync'])[-1]
concur_rtt_async = list(cost_saving['concur_rtt_async'])[-1]
comm_per_access = concur_down_sync + concur_down_async + concur_up_sync + concur_up_async
down_per_batch = concur_down_sync + concur_down_async / 8
up_per_batch = concur_up_sync + concur_up_async / 8


def get_comm_price_from_latency(latency, dec_speed, enc_speed, path_down, path_up, ring_down, ring_up, down_per_batch,
                                up_per_batch):
    path_bw = (path_down + path_up) / (latency - path_down / dec_speed - path_up / enc_speed)
    ring_bw = (ring_down + ring_up) / (latency - ring_down / dec_speed - ring_up / enc_speed)
    concur_bw = (down_per_batch + up_per_batch) / (
            latency - down_per_batch / dec_speed - up_per_batch / enc_speed)
    return get_comm_price(path_bw), get_comm_price(ring_bw) + 0.5168, get_comm_price(concur_bw) + 0.5168


def get_comm_price_from_through(through, dec_speed, enc_speed, path_down, path_up, ring_down, ring_up, down_per_batch,
                                up_per_batch):
    path_bw = (path_down + path_up) / (1 / through - path_down / dec_speed - path_up / enc_speed)
    ring_bw = (ring_down + ring_up) / (1 / through - ring_down / dec_speed - ring_up / enc_speed)
    concur_bw = (down_per_batch + up_per_batch) / (
            8 / through - down_per_batch / dec_speed - up_per_batch / enc_speed)
    return get_comm_price(path_bw), get_comm_price(ring_bw) + 0.5168, get_comm_price(concur_bw) + 0.5168


def get_ORAM_monetary_cost(latency, through):
    dec_speed, enc_speed = single_thread_encryption_speed()
    df = pd.read_csv(config['output_dir'] + 'evict_record_time_comm.csv')
    path_mc = []
    ring_mc = []
    concur_mc = []
    for i in range(len(df['height'])):
        path_down = list(cost_saving['path_down'])[i]
        path_up = list(cost_saving['path_up'])[i]
        ring_down = list(cost_saving['ring_down'])[i]
        ring_up = list(cost_saving['ring_up'])[i]
        concur_down_sync = list(cost_saving['concur_down_sync'])[i]
        concur_down_async = list(cost_saving['concur_down_async'])[i]
        concur_up_sync = list(cost_saving['concur_up_sync'])[i]
        concur_up_async = list(cost_saving['concur_up_async'])[i]
        down_per_batch = concur_down_sync + concur_down_async / 8
        up_per_batch = concur_up_sync + concur_up_async / 8
        tmp1 = get_comm_price_from_latency(latency, dec_speed, enc_speed, path_down, path_up, ring_down, ring_up,
                                           down_per_batch, up_per_batch)
        tmp2 = get_comm_price_from_through(through, dec_speed, enc_speed, path_down, path_up, ring_down, ring_up,
                                           down_per_batch, up_per_batch)
        path_mc.append(max(tmp1[0], tmp2[0]))
        ring_mc.append(max(tmp1[1], tmp2[1]))
        concur_mc.append(max(tmp1[2], tmp2[2]))
    return path_mc, ring_mc, concur_mc, list(df['height'])


def prepare_csv(system_prfm):
    df = DataFrame()

    with tqdm(total=3, ncols=80) as pbar:
        latency_list = [i[0] / 1000 for i in system_prfm]
        throughput_list = [i[1] for i in system_prfm]

        pbar.set_description(f"Calculating # {latency_list[0]} ms {throughput_list[0]} req/s")
        latency = latency_list[0]
        throughput = throughput_list[0]
        set1_path_mc, set1_ring_mc, set1_concur_mc, _ = get_ORAM_monetary_cost(latency, throughput)
        pbar.update(1)

        pbar.set_description(f"Calculating # {latency_list[1]} ms {throughput_list[1]} req/s")
        latency = latency_list[1]
        throughput = throughput_list[1]
        set2_path_mc, set2_ring_mc, set2_concur_mc, _ = get_ORAM_monetary_cost(latency, throughput)
        pbar.update(1)

        pbar.set_description(f"Calculating # {latency_list[2]} ms {throughput_list[2]} req/s")
        latency = latency_list[2]
        throughput = throughput_list[2]
        set3_path_mc, set3_ring_mc, set3_concur_mc, heights = get_ORAM_monetary_cost(latency, throughput)
        pbar.update(1)

    df['set1_path_mc'] = set1_path_mc
    df['set1_ring_mc'] = set1_ring_mc
    df['set1_concur_mc'] = set1_concur_mc
    df['set2_path_mc'] = set2_path_mc
    df['set2_ring_mc'] = set2_ring_mc
    df['set2_concur_mc'] = set2_concur_mc
    df['set3_path_mc'] = set3_path_mc
    df['set3_ring_mc'] = set3_ring_mc
    df['set3_concur_mc'] = set3_concur_mc
    df['block_size'] = heights
    df.to_csv(config['output_dir'] + 'choose_ORAM.csv', index=False)


def draw_figs(fig, axs, system_prfm, csv_path, ignore_legend=False, saveFig=True):
    df = pd.read_csv(csv_path)
    # Common settings for all subplots
    for ax in axs:
        ax.tick_params(axis='x', which='major', pad=5)
        ax.grid(axis='y', color='#e6e6e6', linewidth=2, )
        ax.grid(axis='x', color='#e6e6e6', linewidth=2, )
        ax.set_xscale('log', base=2)
        ax.set_xticks([2 ** 12, 2 ** 16, 2 ** 20])
        ax.set_xlabel('Buckets')

    for ax in axs[1:]:
        ax.tick_params(labelleft=False)

    axs[0].plot(df['block_size'], df['set1_path_mc'], '-o', linewidth=3, color='#ed81b6', markersize=10,
                label='Path')
    axs[0].plot(df['block_size'], df['set1_ring_mc'], '--s', linewidth=3, color='#d33c88', markersize=10,
                label='Ring')
    axs[0].plot(df['block_size'], df['set1_concur_mc'], ':>', linewidth=3, color='#ab2c71', markersize=10,
                label='Concur')
    axs[0].set_title(f'# {system_prfm[0][0]} ms, {system_prfm[0][1]} req/s', fontsize=config['fig_config']['font.size'])
    axs[0].set_ylabel('Monetary cost ($/hr)')

    axs[1].plot(df['block_size'], df['set2_path_mc'], '-o', linewidth=3, color='#ed81b6', markersize=10)
    axs[1].plot(df['block_size'], df['set2_ring_mc'], '--s', linewidth=3, color='#d33c88', markersize=10)
    axs[1].plot(df['block_size'], df['set2_concur_mc'], ':>', linewidth=3, color='#ab2c71', markersize=10)
    axs[1].set_title(f'# {system_prfm[1][0]} ms, {system_prfm[1][1]} req/s', fontsize=config['fig_config']['font.size'])

    axs[2].plot(df['block_size'], df['set3_path_mc'], '-o', linewidth=3, color='#ed81b6', markersize=10)
    axs[2].plot(df['block_size'], df['set3_ring_mc'], '--s', linewidth=3, color='#d33c88', markersize=10)
    axs[2].plot(df['block_size'], df['set3_concur_mc'], ':>', linewidth=3, color='#ab2c71', markersize=10)
    axs[2].set_title(f'# {system_prfm[2][0]} ms, {system_prfm[2][1]} req/s', fontsize=config['fig_config']['font.size'])

    lines_labels = [ax.get_legend_handles_labels() for ax in axs]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    legend = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3,
                        fontsize=config['fig_config']['font.size'], labelspacing=0.25, handletextpad=0.25,
                        columnspacing=1)
    legend.get_frame().set_facecolor('#f7f7f7')
    legend.get_frame().set_edgecolor('#f7f7f7')

    plt.tight_layout(pad=0.2)
    if not ignore_legend:
        plt.subplots_adjust(top=0.75, wspace=0.1)
    else:
        plt.subplots_adjust(top=0.75, wspace=0.1, bottom=0.3)
    if saveFig:
        fig.savefig(config['output_dir'] + 'Fig7_decision_choose_ORAM.pdf')


def main():
    parser = argparse.ArgumentParser(description='Prepare the data used by Figure-11 and draw the resultant figures.')
    parser.add_argument('-p', '--prepare', action='store_true', help='Prepare the data.')
    parser.add_argument('-d', '--draw', action='store_true', help='Draw the resultant figures.')
    parser.add_argument('-c', '--compare', action='store_true',
                        help='Generate comparison with the graphs in the paper.')
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

    system_prfm = [
        [30, 33],
        [20, 33],
        [30, 50],
    ]

    if args.prepare:
        print("# Preparing the data ...")
        prepare_csv(system_prfm)
        print(f"\tData saved to {config['output_dir'] + 'choose_ORAM.csv'}\n")

    if args.draw:
        print("# Drawing the figures ...")
        rcParams.update(config['fig_config'])
        fig, axs = plt.subplots(1, 3, sharey='all', figsize=(9, 4))
        draw_figs(fig, axs, system_prfm, config['output_dir'] + 'choose_ORAM.csv')
        print(f"\tFigure saved to {config['output_dir'] + 'Fig7_decision_choose_ORAM.pdf'}\n")

        if not args.compare:
            plt.show()

    if args.compare:
        print("# Comparing the results with the graphs in the paper ...")
        rcParams.update(config['fig_config'])
        mid = 1.5
        fig = plt.figure(figsize=(18 + mid, 4.5))
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
        draw_figs(fig, axs[:3], system_prfm, config['output_dir'] + 'choose_ORAM.csv', saveFig=False)
        draw_figs(fig, axs[4:], system_prfm, config['paper_data_dir'] + 'choose_ORAM.csv', ignore_legend=True,
                  saveFig=False)
        fig.text(0.25, 0.05, 'Artifact Evaluation', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.text(0.75, 0.05, 'Paper Figure 7', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.savefig(config['output_dir'] + 'Fig7_decision_choose_ORAM_compare.png')
        print(f"\tFigure saved to {config['output_dir'] + 'Fig7_decision_choose_ORAM_compare.png'}\n")
        print("\tThe left is the figures generated by the artifact.\n"
              "\tThe right is the figures in the paper.")
        plt.show()


if __name__ == '__main__':
    main()
