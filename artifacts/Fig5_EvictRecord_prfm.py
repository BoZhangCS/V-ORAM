import os

os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import time
import argparse
import warnings
import pandas as pd
from tqdm import tqdm
from os import urandom
from typing import List
from copy import deepcopy
from random import random, choice
from matplotlib import pyplot as plt, rcParams

from config import default_para as config
from src.ConcurORAM import ConcurORAM
from src.Path_ORAM import Path_ORAM
from src.Ring_ORAM import Ring_ORAM
from src.V_ORAM import V_ORAM
from src.utils import Encoder


def runVORAM(height, **input_config):
    pbar = input_config.get('pbar')
    maxStashSize = input_config.get('maxStashSize')
    block_size = input_config.get('block_size')
    bucket_size = input_config.get('bucket_size')
    s_num = input_config.get('s_num')
    a_num = input_config.get('a_num')
    down_bw = input_config.get('down_bw')
    up_bw = input_config.get('up_bw')
    c_batch = input_config.get('c_batch')
    test_interval = input_config.get('test_interval')
    total_epoch = 2
    workload_period = [('concur', test_interval), ('ring', test_interval)] * total_epoch
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

    list_: List[List] = []
    results = {
        'ring_evicted': deepcopy(list_),
        'ring_amortized': deepcopy(list_),
        'concur_evicted': deepcopy(list_),
        'concur_amortized': deepcopy(list_)
    }

    for sid, access_num in workload_period:
        batch_requests = []
        evict_record_down = 0
        evict_record_up = 0
        evict_access_count = 0
        evict_record_async = [0, 0, 0]

        tmp_down = 0
        tmp_up = 0
        evict_flag = False
        previous_info = deepcopy(v_oram.curr_info)

        size_str = f"{block_size / 1024}KB" if block_size >= 1024 else f"{block_size}B"
        pbar.set_description(f"Height:\t{height}, V-ORAM ({sid.capitalize()}, {size_str})\t")
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
                    _, info = v_oram.access('read', address, urandom(block_size), sid)

                # Ring is measured by a_num access average
                if sid == 'ring':
                    tmp_down += info.down_sync
                    tmp_up += info.up_sync
                    if info.evict_record_flag:
                        evict_flag = True
                    if i % a_num == a_num - 1:
                        if evict_flag:
                            evict_access_count += a_num
                            evict_record_down += tmp_down
                            evict_record_up += tmp_up
                            evict_flag = False
                        tmp_down = 0
                        tmp_up = 0
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

                if info.evict_record_flag:
                    evict_access_count += c_batch
                    evict_record_down += info.down_sync
                    evict_record_up += info.up_sync
                    evict_record_async[0] += info.down_async
                    evict_record_async[1] += info.up_async
            pbar.update(height)

        amortized_down = (v_oram.curr_info.down_sync - previous_info.down_sync) / access_num
        amortized_up = (v_oram.curr_info.up_sync - previous_info.up_sync) / access_num
        amortized_rtt = (v_oram.curr_info.rtt_sync - previous_info.rtt_sync) / access_num
        amortized_async = [(v_oram.curr_info.down_async - previous_info.down_async) / access_num,
                           (v_oram.curr_info.up_async - previous_info.up_async) / access_num,
                           (v_oram.curr_info.rtt_async - previous_info.rtt_async) / access_num]
        if evict_access_count != 0:
            evict_record_down /= evict_access_count
            evict_record_up /= evict_access_count
            evict_record_async[0] /= evict_access_count
            evict_record_async[1] /= evict_access_count

        if sid == 'ring':
            amortized_comm = amortized_down + amortized_up
            amortized_time = amortized_down * (1 / down_bw + 1 / dec_speed) + amortized_up * (1 / up_bw + 1 / enc_speed)

            evicted_comm = evict_record_down + evict_record_up
            evicted_time = evict_record_down * (1 / down_bw + 1 / dec_speed) + evict_record_up * (
                    1 / up_bw + 1 / enc_speed)
            if evict_access_count != 0:
                results['ring_amortized'].append([amortized_time, amortized_comm, amortized_rtt, 0])
                results['ring_evicted'].append([evicted_time, evicted_comm, 0, 0])

        elif sid == 'concur':
            amortized_comm = amortized_down + amortized_up + amortized_async[0] + amortized_async[1]
            amortized_time = (amortized_down + amortized_async[0] / c_batch) * (1 / down_bw + 1 / dec_speed) + (
                    amortized_up + amortized_async[1] / c_batch) * (1 / up_bw + 1 / enc_speed)
            amortized_rtt += amortized_async[2] / c_batch

            evicted_comm = evict_record_down + evict_record_up + evict_record_async[0] + evict_record_async[1]
            evicted_time = (evict_record_down + evict_record_async[0] / c_batch) * (1 / down_bw + 1 / dec_speed) + (
                    evict_record_up + evict_record_async[1] / c_batch) * (1 / up_bw + 1 / enc_speed)
            if evict_access_count != 0:
                results['concur_amortized'].append([amortized_time, amortized_comm, amortized_rtt, amortized_async[-1]])
                results['concur_evicted'].append([evicted_time, evicted_comm, 0, 0])

    for key, value in results.items():
        if len(value) != 0:
            tmp_1 = sum([i[0] for i in value]) / len(value)
            tmp_2 = sum([i[1] for i in value]) / len(value)
            tmp_3 = sum([i[2] for i in value]) / len(value)
            tmp_4 = sum([i[3] for i in value]) / len(value)
            results[key] = [tmp_1, tmp_2, tmp_3, tmp_4]
        else:
            print('No evict record happened')

    return results


def runORAM(ORAM_type, height=10, detail=False, **config):
    pbar = config.get('pbar')
    bucket_size = config.get('bucket_size')
    block_size = config.get('block_size')
    s_num = config.get('s_num')
    a_num = config.get('a_num')
    down_bw = config.get('down_bw')
    up_bw = config.get('up_bw')
    c_batch = config.get('c_batch')
    test_interval = config.get('test_interval')

    encoder = Encoder()
    tmp_cipher = encoder.Enc(urandom(block_size), 'path')
    start_time = time.time()
    for _ in range(test_interval):
        encoder.Dec(tmp_cipher)
    dec_speed = block_size * test_interval / ((time.time() - start_time) / 1000)
    start_time = time.time()
    for _ in range(test_interval):
        encoder.Enc(urandom(block_size), 'path')
    enc_speed = block_size * test_interval / ((time.time() - start_time) / 1000)

    real_datasets = {}
    time_ = 0
    down = 0
    down_sync = 0
    down_async = 0
    up_sync = 0
    up_async = 0
    up = 0
    rtt_sync = 0
    rtt_async = 0
    rtt = 0
    if ORAM_type in ['ring', 'path']:
        if ORAM_type == 'ring':
            oram = Ring_ORAM(height=height, bucket_size=bucket_size, a_num=a_num, s_num=s_num, block_size=block_size)
        else:
            oram = Path_ORAM(height=height, bucket_size=bucket_size, block_size=block_size)
        p_map = oram.position_map

        for i in range(test_interval):
            if random() < 0.5:
                address = choice(list(p_map.keys()))
                data = urandom(block_size)
                _, _, _, info = oram.access('write', address, data)
                real_datasets[address] = data
            else:
                if len(real_datasets) == 0:
                    i -= 1
                    continue
                address = choice(list(real_datasets.keys()))
                _, _, _, info = oram.access('read', address, urandom(block_size))
            down += info.down_sync
            up += info.up_sync
            time_ += info.down_sync / down_bw + info.up_sync / up_bw
            time_ += (info.down_sync / dec_speed + info.up_sync / enc_speed)
            rtt += info.rtt_sync
            pbar.update(height)
    else:
        concur = ConcurORAM(height=height, bucket_size=bucket_size, block_size=block_size,
                            c_batch=c_batch, a_num=a_num, s_num=s_num)
        p_map = concur.position_map

        real_datasets = {}
        for i in range(test_interval // c_batch):
            batch_requests = []
            while len(batch_requests) < c_batch:
                if random() < 0.5:
                    address = choice(list(p_map.keys()))
                    data = urandom(block_size)
                    real_datasets[address] = data
                    batch_requests.append(('write', address, data))
                else:
                    if len(real_datasets) == 0:
                        continue
                    address = choice(list(real_datasets.keys()))
                    real = real_datasets[address]
                    batch_requests.append(('read', address, real))

            results, info = concur.access(batch_requests)
            down_sync += info.down_sync
            down_async += info.down_async
            up_sync += info.up_sync
            up_async += info.up_async
            down += info.down_sync + info.down_async / c_batch
            up += info.up_sync + info.up_async / c_batch
            # communication time cost
            time_ += (info.down_sync + info.down_async / c_batch) / down_bw
            time_ += (info.up_sync + info.up_async / c_batch) / up_bw
            # computation time cost
            time_ += (info.down_sync + info.down_async / c_batch) / dec_speed
            time_ += (info.up_sync + info.up_async / c_batch) / enc_speed
            rtt_sync += info.rtt_sync
            rtt_async += info.rtt_async
            pbar.update(height * c_batch)

    if ORAM_type in ['ring', 'path']:
        return time_ / test_interval, down / test_interval, up / test_interval, rtt / test_interval
    else:
        if not detail:
            return time_ / test_interval, down / test_interval, up / test_interval
        else:
            return (
                time_ / test_interval, down_sync / test_interval, down_async / test_interval, up_sync / test_interval,
                up_async / test_interval, rtt_sync / test_interval, rtt_async / test_interval)


def prepare_csv(height_list):
    df = pd.DataFrame()
    path_time = []
    path_down = []
    path_up = []
    path_comm = []
    path_rtt = []

    ring_time = []
    ring_down = []
    ring_up = []
    ring_comm = []
    ring_rtt = []

    ring_evicted_time = []
    ring_amortized_time = []
    ring_evicted_comm = []
    ring_amortized_comm = []
    ring_amortized_rtt = []

    concur_time = []
    concur_comm = []
    concur_down_sync = []
    concur_down_async = []
    concur_up_sync = []
    concur_up_async = []
    concur_rtt_sync = []
    concur_rtt_async = []

    concur_evicted_time = []
    concur_amortized_time = []
    concur_evicted_comm = []
    concur_amortized_comm = []
    concur_amortized_rtt = []
    concur_amortized_rtt_async = []

    with tqdm(total=7 * sum(height_list) * config['test_interval'], ncols=80) as pbar:
        config['pbar'] = pbar
        for i in height_list:
            pbar.set_description(f"Height:\t{i}, Path ORAM\t\t")
            result = runORAM('path', height=i, **config)
            path_time.append(result[0])
            path_down.append(result[1])
            path_up.append(result[2])
            path_rtt.append(result[3])
            path_comm.append(result[1] + result[2])

            pbar.set_description(f"Height:\t{i}, Ring ORAM\t\t")
            result = runORAM('ring', height=i, **config)
            ring_time.append(result[0])
            ring_down.append(result[1])
            ring_up.append(result[2])
            ring_rtt.append(result[3])
            ring_comm.append(result[1] + result[2])

            pbar.set_description(f"Height:\t{i}, ConcurORAM\t\t")
            result = runORAM('concur', height=i, detail=True, **config)
            concur_time.append(result[0])
            concur_down_sync.append(result[1])
            concur_down_async.append(result[2])
            concur_up_sync.append(result[3])
            concur_up_async.append(result[4])
            concur_rtt_sync.append(result[5])
            concur_rtt_async.append(result[6])
            concur_comm.append(sum(result[1:5]))

            pbar.set_description(f"Height:\t{i}, V-ORAM (Concur)\t")
            result = runVORAM(height=i, **config)
            ring_amortized_time.append(result['ring_amortized'][0])
            concur_amortized_time.append(result['concur_amortized'][0])
            ring_evicted_time.append(result['ring_evicted'][0])
            concur_evicted_time.append(result['concur_evicted'][0])
            ring_amortized_comm.append(result['ring_amortized'][1])
            concur_amortized_comm.append(result['concur_amortized'][1])
            ring_evicted_comm.append(result['ring_evicted'][1])
            concur_evicted_comm.append(result['concur_evicted'][1])
            ring_amortized_rtt.append(result['ring_amortized'][2])
            concur_amortized_rtt.append(result['concur_amortized'][2])
            concur_amortized_rtt_async.append(result['concur_amortized'][3])

    df['height'] = [2 ** i for i in height_list]
    df['path_time'] = path_time
    df['path_comm'] = path_comm
    df['path_down'] = path_down
    df['path_up'] = path_up
    df['path_rtt'] = path_rtt

    df['ring_time'] = ring_time
    df['ring_comm'] = ring_comm
    df['ring_down'] = ring_down
    df['ring_up'] = ring_up
    df['ring_rtt'] = ring_rtt
    df['ring_amortized_time'] = ring_amortized_time
    df['ring_evicted_time'] = ring_evicted_time
    df['ring_amortized_comm'] = ring_amortized_comm
    df['ring_evicted_comm'] = ring_evicted_comm
    df['ring_amortized_rtt'] = ring_amortized_rtt

    df['concur_time'] = concur_time
    df['concur_comm'] = concur_comm
    df['concur_down_sync'] = concur_down_sync
    df['concur_down_async'] = concur_down_async
    df['concur_up_sync'] = concur_up_sync
    df['concur_up_async'] = concur_up_async
    df['concur_rtt_sync'] = concur_rtt_sync
    df['concur_rtt_async'] = concur_rtt_async
    df['concur_amortized_time'] = concur_amortized_time
    df['concur_evicted_time'] = concur_evicted_time
    df['concur_amortized_comm'] = concur_amortized_comm
    df['concur_evicted_comm'] = concur_evicted_comm
    df['concur_amortized_rtt'] = concur_amortized_rtt
    df['concur_amortized_rtt_async'] = concur_amortized_rtt_async

    df.to_csv(config['output_dir'] + 'evict_record_time_comm.csv', index=False)


def draw_figs(fig, axs, csv_path, ignore_summary=False, ignore_legend=False, saveFig=True):
    markersize = 10
    top = 0.77

    df = pd.read_csv(csv_path)
    for ax_ in axs:
        for ax in ax_:
            ax.tick_params(axis='x', which='major', pad=5)
            ax.grid(axis='y', color='#e6e6e6', linewidth=2, )
            ax.grid(axis='x', color='#e6e6e6', linewidth=2, )
            ax.set_xscale('log', base=2)
            ax.set_xticks([2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20])
    axs[1][0].set_xlabel('Buckets')
    axs[1][1].set_xlabel('Buckets')

    axs[0][0].set_ylim([20, 50])
    axs[0][0].set_yticks([25, 45])
    axs[0][0].plot(df['height'], 1000 * df['ring_evicted_time'], '-o', linewidth=3, color='#91caff',
                   markersize=markersize,
                   label='Ring EvictRecord')
    axs[0][0].plot(df['height'], 1000 * df['ring_amortized_time'], '-s', linewidth=3, color='#4096ff',
                   markersize=markersize,
                   label='Ring amortized')
    axs[0][0].plot(df['height'], 1000 * df['ring_time'], '-^', linewidth=3, color='#0958d9', markersize=markersize,
                   label='Ring original')

    axs[1][0].set_ylim([15, 30])
    axs[1][0].set_yticks([20, 25])
    axs[1][0].plot(df['height'], 1000 * df['concur_evicted_time'], ':o', linewidth=3, color='#ff85c0',
                   markersize=markersize,
                   label='Concur EvictRecord')
    axs[1][0].plot(df['height'], 1000 * df['concur_amortized_time'], ':s', linewidth=3, color='#eb2f96',
                   markersize=markersize,
                   label='Concur amortized')
    axs[1][0].plot(df['height'], 1000 * df['concur_time'], ':^', linewidth=3, color='#c41d7f', markersize=markersize,
                   label='Concur original')
    axs[1][0].set_ylabel('Processing time (ms)')
    axs[1][0].yaxis.set_label_coords(-0.14, 1)

    axs[0][1].set_ylim([200, 500])
    axs[0][1].set_yticks([250, 450])
    axs[0][1].plot(df['height'], df['ring_evicted_comm'] / 2 ** 10, '-o', linewidth=3, color='#91caff',
                   markersize=markersize)
    axs[0][1].plot(df['height'], df['ring_amortized_comm'] / 2 ** 10, '-s', linewidth=3, color='#4096ff',
                   markersize=markersize)
    axs[0][1].plot(df['height'], df['ring_comm'] / 2 ** 10, '-^', linewidth=3, color='#0958d9', markersize=markersize)

    axs[1][1].set_ylim([900, 1050])
    axs[1][1].set_yticks([950, 1000])
    axs[1][1].plot(df['height'], df['concur_comm'] / 2 ** 10, ':^', linewidth=3, color='#c41d7f',
                   markersize=markersize)
    axs[1][1].plot(df['height'], df['concur_evicted_comm'] / 2 ** 10, ':o', linewidth=3, color='#ff85c0',
                   markersize=markersize)
    axs[1][1].plot(df['height'], df['concur_amortized_comm'] / 2 ** 10, ':s', linewidth=3, color='#eb2f96',
                   markersize=markersize)
    axs[1][1].set_ylabel('Comm. cost (KB)')
    axs[1][1].yaxis.set_label_coords(-0.22, 1)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines = [lines[i] for i in [0, 3, 1, 4, 2, 5]]
    labels = [labels[i] for i in [0, 3, 1, 4, 2, 5]]

    if not ignore_legend:
        ncol = 3 if not ignore_summary else 6
        legend = fig.legend(lines, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, 1),
                            fontsize=config['fig_config']['font.size'] * 0.88, labelspacing=0.25, handletextpad=0.25,
                            columnspacing=0.6)
        legend.get_frame().set_facecolor('#f7f7f7')
        legend.get_frame().set_edgecolor('#f7f7f7')

    plt.subplots_adjust(wspace=-.25)
    plt.tight_layout(pad=0.3)

    if not ignore_legend:
        plt.subplots_adjust(top=top)
    else:
        plt.subplots_adjust(top=top * 1.1, bottom=0.3)
    if saveFig:
        fig.savefig(config['output_dir'] + 'Fig5_EvictRecord_prfm.pdf')

    ring_evicted_time_diff = abs(df['ring_evicted_time'] - df['ring_time']).max()
    ring_amortized_time_diff = abs(df['ring_amortized_time'] - df['ring_time']).max()
    concur_evicted_time_diff = abs(df['concur_evicted_time'] - df['concur_time']).max()
    concur_amortized_time_diff = abs(df['concur_amortized_time'] - df['concur_time']).max()
    ring_evicted_comm_diff = abs(df['ring_evicted_comm'] - df['ring_comm']).max()
    ring_amortized_comm_diff = abs(df['ring_amortized_comm'] - df['ring_comm']).max()
    concur_evicted_comm_diff = abs(df['concur_evicted_comm'] - df['concur_comm']).max()
    concur_amortized_comm_diff = abs(df['concur_amortized_comm'] - df['concur_comm']).max()

    if (not ignore_legend) and (not ignore_summary):
        print(f"Brief summary:")
        print(f"\tFor Ring ORAM:")
        print(f"\tMax diff of Evicted and Original:"
              f"\t{ring_evicted_time_diff * 1000:.2f} ms,\t{ring_evicted_comm_diff / 1024:.2f} kB")
        print(f"\tMax diff of Amortized and Original:"
              f"\t{ring_amortized_time_diff * 1000:.2f} ms,\t{ring_amortized_comm_diff / 1024:.2f} kB")
        print(f"\tMax diff of Evicted and Original:"
              f"\t{concur_evicted_time_diff * 1000:.2f} ms,\t{concur_evicted_comm_diff / 1024:.2f} kB")
        print(f"\tMax diff of Amortized and Original:"
              f"\t{concur_amortized_time_diff * 1000:.2f} ms,\t{concur_amortized_comm_diff / 1024:.2f} kB\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare the data used by Figure-11 and draw the resultant figures.')
    parser.add_argument('-p', '--prepare', action='store_true', help='Prepare the data.')
    parser.add_argument('-d', '--draw', action='store_true', help='Draw the resultant figures.')
    parser.add_argument('-c', '--compare', action='store_true',
                        help='Generate comparison with the graphs in the paper.')
    parser.add_argument('-s', '--start', type=int, default=12, help='Start height (default: 12)')
    parser.add_argument('-e', '--end', type=int, default=20, help='End height (default: 20)')
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

    if args.prepare:
        print("# Preparing the data ...")
        start_height = args.start
        end_height = args.end
        prepare_csv(list(range(start_height, end_height + 1, 2)))
        print(f"\tData saved to {config['output_dir'] + 'evict_record_time_comm.csv'}\n")

    if args.draw:
        print("# Drawing the figures ...")
        rcParams.update(config['fig_config'])
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all', figsize=(8.4, 4.286))
        draw_figs(fig, axs, config['output_dir'] + 'evict_record_time_comm.csv')
        print(f"\tFigure saved to {config['output_dir'] + 'Fig5_EvictRecord_prfm.pdf'}\n")

        if not args.compare:
            plt.show()

    if args.compare:
        print("# Comparing the results with the graphs in the paper ...")
        rcParams.update(config['fig_config'])
        fig, axs = plt.subplots(nrows=2, ncols=4, sharex='all', figsize=(8.4 * 2.1, 4.5))
        draw_figs(fig, [axs[0][:2], axs[1][:2]], config['output_dir'] + 'evict_record_time_comm.csv',
                  ignore_summary=True, saveFig=False)
        draw_figs(fig, [axs[0][2:], axs[1][2:]], config['paper_data_dir'] + 'evict_record_time_comm.csv',
                  ignore_legend=True, saveFig=False)
        fig.text(0.25, 0.05, 'Artifact Evaluation', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.text(0.75, 0.05, 'Paper Figure 5', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.savefig(config['output_dir'] + 'Fig5_EvictRecord_prfm_compare.png')
        print(f"\tFigure saved to {config['output_dir'] + 'Fig5_EvictRecord_prfm_compare.png'}\n")
        print("\tThe left is the figures generated by the artifact.\n"
              "\tThe right is the figures in the paper.")
        plt.show()


if __name__ == '__main__':
    main()
