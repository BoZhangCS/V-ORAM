import argparse
import os
import warnings

from tqdm import tqdm

os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import time
import pandas as pd
from os import urandom
from pandas import DataFrame
from random import choice, random

from src.utils import Encoder
from src.V_ORAM import V_ORAM
from config import default_para as config
from matplotlib import pyplot as plt, rcParams, patches, gridspec


def get_rtt():
    return 0
    # return np.random.normal(0.05, 0.005, 1)


def getWorkload(workload_period, height=10, **input_config):
    maxStashSize = input_config.get('maxStashSize')
    block_size = input_config.get('block_size')
    bucket_size = input_config.get('bucket_size')
    s_num = input_config.get('s_num')
    a_num = input_config.get('a_num')
    down_bw = input_config.get('down_bw')
    up_bw = input_config.get('up_bw')
    pbar = input_config.get('pbar')
    c_batch = a_num
    v_oram = V_ORAM(height=height, bucket_size=bucket_size, block_size=block_size, s_num=s_num, a_num=a_num,
                    c_batch=c_batch, maxStashSize=maxStashSize)

    real_datasets = {}
    p_map = v_oram.curr_ORAM.position_map
    encoder = Encoder()
    tmp_cipher = encoder.Enc(urandom(block_size), 'path')
    test_factor = 10
    start_time = time.time()
    for _ in range(2 ** test_factor):
        encoder.Dec(tmp_cipher)
    dec_speed = block_size * 2 ** test_factor / ((time.time() - start_time) / 1000)
    start_time = time.time()
    for _ in range(2 ** test_factor):
        encoder.Enc(urandom(block_size), 'path')
    enc_speed = block_size * 2 ** test_factor / ((time.time() - start_time) / 1000)

    result_throughput = []
    result_latency = []
    total_comm = 0
    total_time = 0
    for sid, access_num in workload_period:
        pbar.set_description(f'Height: {height}, V-ORAM ({sid.capitalize()})\t')
        batch_requests = []
        tmp_time = 0
        for i in range(access_num):
            if sid != 'concur':
                if random() < 0.5:
                    address = choice(list(p_map.keys()))
                    data = urandom(block_size)
                    _, info = v_oram.access('write', address, data, sid)
                    real_datasets[address] = data
                else:
                    if len(real_datasets) == 0:
                        i -= 1
                        continue
                    address = choice(list(real_datasets.keys()))
                    _, info = v_oram.access('read', address, urandom(4096), sid)

                if sid == 'ring':
                    total_comm += info.down_sync + info.up_sync
                    tmp_time += info.down_sync / down_bw + info.up_sync / up_bw + info.rtt_sync * get_rtt()
                    tmp_time += (info.down_sync / dec_speed + info.up_sync / enc_speed)
                    if i % a_num == a_num - 1:
                        result_throughput += [a_num / tmp_time for _ in range(a_num)]
                        result_latency += [tmp_time / a_num for _ in range(a_num)]
                        total_time += tmp_time
                        tmp_time = 0
                else:
                    total_comm += info.down_sync + info.up_sync
                    tmp_time = info.down_sync / down_bw + info.up_sync / up_bw + info.rtt_sync * get_rtt()
                    tmp_time += (info.down_sync / dec_speed + info.up_sync / enc_speed)
                    total_time += tmp_time
                    result_throughput.append(1 / tmp_time)
                    result_latency.append((tmp_time))
            else:
                # Concur is measured by c_batch ** 2 access average
                if random() < 0.5:
                    address = choice(list(p_map.keys()))
                    data = urandom(block_size)
                    real_datasets[address] = data
                    batch_requests.append(('write', address, data))
                    _, info = v_oram.access('write', address, data, 'concur')
                else:
                    if len(real_datasets) == 0:
                        continue
                    address = choice(list(real_datasets.keys()))
                    real = real_datasets[address]
                    dummy = urandom(block_size)
                    batch_requests.append(('read', address, real))
                    _, info = v_oram.access('read', address, dummy, 'concur')

                if i == access_num - 1 and len(batch_requests) < c_batch:
                    while len(batch_requests) < c_batch:
                        address = choice(list(real_datasets.keys()))
                        real = real_datasets[address]
                        dummy = urandom(block_size)
                        batch_requests.append(('read', address, real))
                        _, info = v_oram.access('read', address, dummy, 'concur')

                if len(batch_requests) == c_batch:
                    batch_requests = []

                if info is not None and i % c_batch == c_batch - 1:
                    total_comm += info.down_sync + info.down_async + info.up_sync + info.up_async
                    # rtt time
                    tmp_time += (info.rtt_sync + info.rtt_async / c_batch) * get_rtt()
                    # communication time cost
                    tmp_time += (info.down_sync + info.down_async / c_batch) / down_bw
                    tmp_time += (info.up_sync + info.up_async / c_batch) / up_bw
                    # computation time cost
                    tmp_time += (info.down_sync + info.down_async / c_batch) / dec_speed
                    tmp_time += (info.up_sync + info.up_async / c_batch) / enc_speed

                    result_throughput += [c_batch / tmp_time for _ in range(c_batch)]
                    result_latency += [tmp_time for _ in range(c_batch)]
                    total_time += tmp_time
                    tmp_time = 0
            pbar.update(height)

    return result_throughput, result_latency, total_comm, total_time


def prepare_csv():
    df = DataFrame()
    total = 0
    for latency, throughput in zip(latency_workload, throughput_workload):
        total += latency[1] + throughput[1]
    with tqdm(total=total * 12, ncols=80) as pbar:
        config['pbar'] = pbar
        latency_list = getWorkload(latency_workload, 12, **config)[1]
        throughput_list = getWorkload(throughput_workload, 12, **config)[0]
        min_len = min(len(latency_list), len(throughput_list))
        df['latency'] = latency_list[:min_len]
        df['through'] = throughput_list[:min_len]
        for i in range(len(df['latency'])):
            tmp1 = tmp2 = 0
            len_ = min(100, len(df['latency']) - i)
            for j in range(i, i + len_):
                tmp1 += df['through'][j]
                tmp2 += df['latency'][j]
            df['through'][i] = tmp1 / len_
            df['latency'][i] = tmp2 / len_
    df.to_csv(config['output_dir'] + 'simulated_workloads.csv', index=False)


def draw_figs(fig, axs, csv_path, ignore_legend=False, saveFig=True):
    global interval
    df = pd.read_csv(csv_path)
    latency_list = df['latency']
    throughput_list = df['through']
    if ignore_legend:  # The paper uses test intervals of 2^11
        interval = config['test_interval'] * 2

    axs[0].plot(throughput_list, color='#9370db', linewidth=3)
    rect = patches.Rectangle((4 * interval, 8), 2 * interval, 30, linewidth=3, edgecolor='r', facecolor='none',
                             zorder=10)
    axs[0].add_patch(rect)
    axs[0].set_ylim([0, 60])
    axs[0].text(0.08, 0.95, 'Path', transform=axs[0].transAxes, fontsize=22, color='gray', verticalalignment='top')
    axs[0].text(0.34, 0.95, 'Concur', transform=axs[0].transAxes, fontsize=22, color='gray', verticalalignment='top')
    axs[0].set_ylabel('Throughput (req/s)')
    axs[0].set_title('Dynamic Throughput', fontsize=config['fig_config']['font.size'])

    axs[1].plot(latency_list * 1000, color='#9370db', linewidth=3)
    rect = patches.Rectangle((4 * interval, 30), 2 * interval, 95, linewidth=3, edgecolor='r', facecolor='none',
                             zorder=10)
    axs[1].add_patch(rect)
    axs[1].set_ylim([0, 150])
    axs[1].text(0.08, 0.95, 'Path', transform=axs[1].transAxes, fontsize=22, color='gray', verticalalignment='top')
    axs[1].text(0.32, 0.95, 'Ring', transform=axs[1].transAxes, fontsize=22, color='gray', verticalalignment='top')
    axs[1].set_ylabel('Latency (ms)')  # Change the label to 'Latency (ms)'
    axs[1].set_title('Dynamic Latency', fontsize=config['fig_config']['font.size'])

    for ax in axs:
        ax.tick_params(axis='x', which='major', pad=5)
        ax.grid(axis='y', color='#e6e6e6', linewidth=2, zorder=-1)
        ax.grid(axis='x', color='#e6e6e6', linewidth=2, zorder=-1)
        ax.set_xticks([0, interval * 4, interval * 8, interval * 12, interval * 16], [0, 4, 8, 12, 16])
        ax.axvspan(5 * interval, 8 * interval, facecolor='g', alpha=0.10)
        ax.axvspan(13 * interval, 16 * interval, facecolor='g', alpha=0.10)
        ax.set_xlabel('Interval index')

    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout(pad=0.2)
    if ignore_legend:
        plt.subplots_adjust(wspace=0.5, bottom=0.35)
    if saveFig:
        fig.savefig(config['output_dir'] + 'Fig11_simulated_workloads.pdf')


def main():
    parser = argparse.ArgumentParser(description='Prepare the data used by Figure-11 and draw the resultant figures.')
    parser.add_argument('-p', '--prepare', action='store_true', help='Prepare the data.')
    parser.add_argument('-d', '--draw', action='store_true', help='Draw the resultant figures.')
    parser.add_argument('-c', '--compare', action='store_true',
                        help='Generate comparison with the graphs in the paper.')
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

    if args.prepare:
        print("# Preparing the data ...")
        prepare_csv()
        print(f"\tData saved to {config['output_dir'] + 'simulated_workloads.csv'}\n")

    if args.draw:
        print("# Drawing the figures ...")
        rcParams.update(config['fig_config'])
        fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.24))
        draw_figs(fig, axs, config['output_dir'] + 'simulated_workloads.csv')
        print(f"\tFigure saved to {config['output_dir'] + 'Fig11_simulated_workloads.pdf'}\n")

        if not args.compare:
            plt.show()

    if args.compare:
        print("# Comparing the results with the graphs in the paper ...")
        rcParams.update(config['fig_config'])

        mid = 0
        fig = plt.figure(figsize=(8.3 * 2 + mid, 3.24 * 1.2))
        gs = gridspec.GridSpec(1, 5, figure=fig, width_ratios=[4.2, 4.2, mid, 4.2, 4.2])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax_mid = fig.add_subplot(gs[0, 2])
        ax_mid.axis('off')
        ax3 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[0, 4])

        axs = [ax1, ax2, ax3, ax4]

        draw_figs(fig, axs[:2], config['output_dir'] + 'simulated_workloads.csv', saveFig=False)
        draw_figs(fig, axs[2:], config['paper_data_dir'] + 'simulated_workloads.csv',
                  ignore_legend=True, saveFig=False)
        fig.text(0.25, 0.05, 'Artifact Evaluation', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.text(0.75, 0.05, 'Paper Figure 11', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.savefig(config['output_dir'] + 'Fig11_simulated_workloads_compare.png')
        print(f"\tFigure saved to {config['output_dir'] + 'Fig11_simulated_workloads_compare.png'}\n")
        print("\tThe left is the figures generated by the artifact.\n"
              "\tThe right is the figures in the paper.")
        plt.show()


if __name__ == '__main__':
    interval = config['test_interval'] * 2
    total_epoch = 2
    latency_workload = [('path', 5 * interval), ('ring', 3 * interval)] * total_epoch
    throughput_workload = [('path', 5 * interval), ('concur', 3 * interval)] * total_epoch
    main()
