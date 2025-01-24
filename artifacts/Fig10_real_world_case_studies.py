import os

os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import argparse
import warnings

import numpy as np
import pandas as pd
from math import log2, ceil
import matplotlib.pyplot as plt
from matplotlib import rcParams
from config import default_para as config
from artifacts.Fig7_decision_choose_ORAM import single_thread_encryption_speed, get_comm_price

file_dirs = {
    "msrc": config['datasets_dir'] + 'MSRC/src1_0_tripped.csv',
    "twitter": config['datasets_dir'] + 'Twitter/cluster11.2_tripped.csv',
    "alicloud": config['datasets_dir'] + 'AliCloud/io_traces_32.csv'
}

column_names = {
    "msrc": ['Timestamp', 'Hostname', 'DiskNumber', 'Type', 'Offset', 'Size', 'ResponseTime'],
    "twitter": ['Timestamp', 'Key', 'KeySize', 'Size', 'ClientID', 'Type', 'TTL'],
    "alicloud": ['DeviceID', 'Type', 'Offset', 'Size', 'Timestamp']
}

time_units = {
    "msrc": 1e7,
    "twitter": 1,
    "alicloud": 1e6,
}


def get_throughput(workload, window_num, max_lines=None):
    if max_lines:
        df = pd.read_csv(file_dirs[workload], low_memory=False, nrows=max_lines)
    else:
        df = pd.read_csv(file_dirs[workload], low_memory=False)

    new_column_names = column_names[workload]
    df.columns = new_column_names

    if workload != "twitter":
        max_data_size = df['Offset'].max() / 2 ** 30
        print(f'\t\tMax database size:\t{max_data_size:.2f}GB, tree height:\t{ceil(log2(max_data_size) + 16)}')
    else:
        max_data_size = 267.71
        print(
            f'\t\tWe use average cluster size in Twitter"s paper. \n'
            f'\t\tMax database size: 267.71GB, tree height:\t{ceil(log2(max_data_size) + 16)}')

    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    if workload != "twitter":
        window_size = len(df) // window_num
        time_unit = time_units[workload]
        df['TimeDiff'] = df['Timestamp'].diff(periods=window_size)
        df['RequestThroughput'] = window_size / df['TimeDiff'] * time_unit
        df['DataThroughput'] = df['Size'].rolling(window=window_size).sum() / df['TimeDiff']

        timestamp_list = list((df['Timestamp'][window_size - 1:] - df['Timestamp'][window_size - 1]) / time_unit)
        req_tput = df['RequestThroughput'][window_size - 1:].tolist()
    else:
        value_counts = df['Timestamp'].value_counts().sort_index()
        values = value_counts.index.tolist()
        counts = value_counts.values.tolist()
        timestamp_list = df['Timestamp'].tolist()

        data = {'value': values, 'count': counts}
        df = pd.DataFrame(data)
        window_size = len(df) // window_num
        # Calculate the rolling average for the 'count' column
        df['count_avg'] = df['count'].rolling(window=window_size, min_periods=1).mean()
        req_tput = df['count_avg'].tolist()
    return timestamp_list, req_tput, ceil(log2(max_data_size) + 16)


def get_ORAM_comm_cost(height):
    df = pd.read_csv(config['output_dir'] + 'evict_record_time_comm.csv')
    heights = [int(log2(height)) for height in list(df['height'])]
    path_para = np.polyfit(heights, df['path_comm'], 1)
    ring_para = np.polyfit(heights, df['ring_amortized_comm'], 1)
    return path_para[0] * height + path_para[1], ring_para[0] * height + ring_para[1]


def get_monetary_savings(tree_height, sim_low, sim_high):
    comp_num = 1
    base_comp_cost = 0.544  # AWS c6g.4xlarge
    tmp_dec, tmp_enc = single_thread_encryption_speed()
    path_comm, ring_comm = get_ORAM_comm_cost(tree_height)
    ring_down = (ring_comm + 2048) / 2
    ring_up = (ring_comm - 2048) / 2

    dec_speed = tmp_dec
    enc_speed = tmp_enc
    while (1 / sim_high - path_comm / 2 / dec_speed - path_comm / 2 / enc_speed < 0 or
           1 / sim_low - ring_down / 2 / dec_speed - ring_up / 2 / enc_speed < 0):
        comp_num += 1
        dec_speed += tmp_dec
        enc_speed += tmp_enc

    path_low_bw = path_comm / (1 / sim_low - path_comm / 2 / tmp_dec - path_comm / 2 / tmp_enc)
    path_high_bw = path_comm / (1 / sim_high - path_comm / 2 / tmp_dec - path_comm / 2 / tmp_enc)
    ring_low_bw = ring_comm / (1 / sim_low - ring_down / 2 / tmp_dec - ring_up / 2 / tmp_enc)
    ring_high_bw = ring_comm / (1 / sim_high - ring_down / 2 / tmp_dec - ring_up / 2 / tmp_enc)

    comp_price = comp_num * base_comp_cost

    return (get_comm_price(path_low_bw) + comp_price, get_comm_price(path_high_bw) + comp_price,
            get_comm_price(ring_low_bw) + comp_price, get_comm_price(ring_high_bw) + comp_price)


def draw_figs(fig, axs):
    for ax_ in axs:
        for ax in ax_:
            ax.tick_params(axis='x', which='major', pad=5)
            ax.grid(axis='y', color='#e6e6e6', linewidth=2, )
            ax.grid(axis='x', color='#e6e6e6', linewidth=2, )
    for ax in axs[:, 0]:
        ax.sharex(axs[0, 0])
    for ax in axs[:, 1]:
        ax.sharex(axs[0, 1])
    for ax in axs[0, :2]:
        ax.sharey(axs[0, 0])
    for ax in axs[1, :2]:
        ax.sharey(axs[1, 0])

    axs[0, 0].set_ylim(-7e3 / 30, 8e3)
    axs[0, 0].set_yticks([0, 2e3, 4e3, 6e3])
    axs[0, 0].set_yticklabels(['0', '$2k$', '$4k$', '$6k$'])
    axs[1, 0].set_ylim([-3, 90])
    axs[1, 0].set_yticks([0, 30, 60, 90])

    axs[1, 0].set_xlabel('Request index')
    axs[1, 1].set_xlabel('Request index')
    axs[1, 2].set_xlabel('Request index')
    axs[0, 0].set_ylabel('Throughput (req/s)')
    axs[1, 0].set_ylabel('Monetary cost ($/hr)')

    workload = 'msrc'
    print(f'Brief summary:')
    print(f'\t## MSRC:')

    timestamp_list, req_tput, tree_height = get_throughput(workload, 50, 10000)

    sim_low = 1200
    sim_high = 6500

    tmp = get_monetary_savings(tree_height, sim_low, sim_high)
    original_ring_high = tmp[3]
    strategy_average = tmp[0] * 0.76 + tmp[3] * 0.24

    print(f'\t\tWorkload time span:\t{timestamp_list[-1] / 3600:.2f} hr')
    print(f'\t\tRing ORAM monetary costs: {original_ring_high:.2f} $/hr, V-ORAM average: {strategy_average:.2f} $/hr')
    print(f'\t\tSaving:\t{(original_ring_high - strategy_average) * 100 / original_ring_high:.2f} % cost\n')

    sim_pair = [(sim_low, 700), (sim_high, 1200), (sim_low, 4300), (sim_high, 1200), (sim_low, 2500)]
    sim_list = [[tput for _ in range(interval)] for tput, interval in sim_pair]
    sim_tput = [item for sublist in sim_list for item in sublist]

    sim_mc = []
    sim_mc += [tmp[0]] * 700
    sim_mc += [tmp[3]] * 1200
    sim_mc += [tmp[0]] * 4300
    sim_mc += [tmp[3]] * 1200
    sim_mc += [tmp[0]] * 2500

    axs[0, 0].plot(req_tput, color='#9370db', linewidth=3, label='Real wkld.')
    axs[0, 0].plot(sim_tput, color='#5388f8', linestyle='--', linewidth=3, label='Estab. wkld.')
    axs[0, 0].text(0.08, 0.97, 'Ring', transform=axs[0, 0].transAxes, fontsize=config['fig_config']['font.size'],
                   color='gray',
                   verticalalignment='top')
    axs[0, 0].text(0.32, 0.3, 'Path', transform=axs[0, 0].transAxes, fontsize=config['fig_config']['font.size'],
                   color='gray',
                   verticalalignment='top')
    axs[0, 0].set_title('# MRSC $src$1-0', fontsize=config['fig_config']['font.size'])
    axs[0, 0].axvspan(700, 1900, facecolor='g', alpha=0.10)
    axs[0, 0].axvspan(6200, 7400, facecolor='g', alpha=0.10)
    axs[0, 0].set_xticks([0, 5e3, 1e4])
    axs[0, 0].set_xticklabels(['$0$', '$5k$', '$10k$'])

    get_ORAM_comm_cost(24)
    axs[1, 0].plot(sim_mc, color='#3c8990', linestyle='-.', linewidth=3, label='Estab. monetary cost')
    axs[1, 0].plot([tmp[3]] * len(sim_mc), color='#eb2f96', linestyle='-', linewidth=3, label="Ring monetary cost")
    axs[1, 0].set_xticks([0, 5e3, 1e4])
    axs[1, 0].axvspan(700, 1900, facecolor='g', alpha=0.10)
    axs[1, 0].axvspan(6200, 7400, facecolor='g', alpha=0.10)
    axs[1, 0].set_xticklabels(['$0$', '$5k$', '$10k$'])
    axs[1, 0].set_title(f'# Established $N=2^{{{tree_height}}}$', fontsize=config['fig_config']['font.size'])

    workload = 'alicloud'
    print(f'\t## AliCloud:')
    timestamp_list, req_tput, tree_height = get_throughput(workload, 300, 20000000)

    sim_low = 1000
    sim_high = 6000

    tmp = get_monetary_savings(tree_height, sim_low, sim_high)
    original_ring_high = tmp[3]
    strategy_average = tmp[0] * 0.5 + tmp[3] * 0.5

    print(f'\t\tWorkload time span:\t{timestamp_list[-1] / 3600:.2f} hr, {timestamp_list[-1]:.2f} s')
    print(f'\t\tRing ORAM monetary costs: {original_ring_high:.2f} $/hr, V-ORAM average: {strategy_average:.2f} $/hr')
    print(f'\t\tSaving:\t{(original_ring_high - strategy_average) * 100 / original_ring_high:.2f} % cost\n')

    tmp_spots = [100, 1150, 420, 950, 772]
    spots = [int(x * 1e6 / 2912) for x in tmp_spots]

    sim_pair = [(sim_low, spots[0]), (sim_high, spots[1]), (sim_low, spots[2]), (sim_high, spots[3]),
                (sim_low, spots[4])]
    sim_list = [[tput for _ in range(interval)] for tput, interval in sim_pair]
    sim_tput = [item for sublist in sim_list for item in sublist]

    sim_mc = []
    sim_mc += [tmp[0]] * spots[0]
    sim_mc += [tmp[3]] * spots[1]
    sim_mc += [tmp[0]] * spots[2]
    sim_mc += [tmp[3]] * spots[3]
    sim_mc += [tmp[0]] * spots[4]

    axs[0, 1].plot(req_tput, color='#9370db', linewidth=3)
    axs[0, 1].plot(sim_tput, color='#5388f8', linestyle='--', linewidth=3)
    axs[0, 1].set_title('# AliCloud $device$-32', fontsize=config['fig_config']['font.size'])
    axs[0, 1].axvspan(spots[0], sum(spots[:2]), facecolor='g', alpha=0.10)
    axs[0, 1].axvspan(sum(spots[:3]), sum(spots[:4]), facecolor='g', alpha=0.10)
    axs[0, 1].text(0.12, 0.95, 'Ring', transform=axs[0, 1].transAxes, fontsize=config['fig_config']['font.size'],
                   color='gray',
                   verticalalignment='top')
    axs[0, 1].text(0.77, 0.3, 'Path', transform=axs[0, 1].transAxes, fontsize=config['fig_config']['font.size'],
                   color='gray',
                   verticalalignment='top')
    axs[0, 1].set_xticks([0, 5e5, 1e6])
    axs[0, 1].set_xticklabels(['$0$', '$500k$', '$1m$'])

    axs[1, 1].plot(sim_mc, color='#3c8990', linestyle='-.', linewidth=3)
    axs[1, 1].plot([tmp[3]] * len(sim_mc), color='#eb2f96', linestyle='-', linewidth=3)
    axs[1, 1].axvspan(spots[0], sum(spots[:2]), facecolor='g', alpha=0.10)
    axs[1, 1].axvspan(sum(spots[:3]), sum(spots[:4]), facecolor='g', alpha=0.10)
    axs[1, 1].set_xticks([0, 5e5, 1e6])
    axs[1, 1].set_xticklabels(['$0$', '$500k$', '$1m$'])
    axs[1, 1].set_title(f'# Established $N=2^{{{tree_height}}}$', fontsize=config['fig_config']['font.size'])

    workload = 'twitter'
    print(f'\t## Twitter:')
    timestamp_list, req_tput, tree_height = get_throughput(workload, 50, 3000000)

    sim_low = 7800
    sim_high = 9750

    tmp = get_monetary_savings(tree_height, sim_low, sim_high)
    original_ring_high = tmp[3]
    strategy_average = tmp[0] * 0.5 + tmp[3] * 0.5

    print(
        f'\t\tWorkload time span:\t{(timestamp_list[-1] - timestamp_list[0]) / 3600:.2f} hr, {(timestamp_list[-1] - timestamp_list[0]):.2f} s')
    print(f'\t\tRing ORAM monetary costs: {original_ring_high:.2f} $/hr, V-ORAM average: {strategy_average:.2f} $/hr')
    print(f'\t\tSaving:\t{(original_ring_high - strategy_average) * 100 / original_ring_high:.2f} % cost\n')

    spots = [25, 50, 250, 50, 98]

    sim_pair = [(sim_low, spots[0]), (sim_high, spots[1]), (sim_low, spots[2]), (sim_high, spots[3]),
                (sim_low, spots[4])]
    sim_list = [[tput for _ in range(interval)] for tput, interval in sim_pair]
    sim_tput = [item for sublist in sim_list for item in sublist]

    sim_mc = []
    sim_mc += [tmp[0]] * spots[0]
    sim_mc += [tmp[3]] * spots[1]
    sim_mc += [tmp[0]] * spots[2]
    sim_mc += [tmp[3]] * spots[3]
    sim_mc += [tmp[0]] * spots[4]

    axs[0, 2].plot(req_tput, color='#9370db', linewidth=3)
    axs[0, 2].plot(sim_tput, color='#5388f8', linestyle='--', linewidth=3)
    axs[0, 2].set_title('# Twitter $cluster$-11.2', fontsize=config['fig_config']['font.size'])
    axs[0, 2].axvspan(spots[0], sum(spots[:2]), facecolor='g', alpha=0.10)
    axs[0, 2].axvspan(sum(spots[:3]), sum(spots[:4]), facecolor='g', alpha=0.10)
    axs[0, 2].text(0.05, 0.95, 'Ring', transform=axs[0, 2].transAxes, fontsize=config['fig_config']['font.size'],
                   color='gray',
                   verticalalignment='top')
    axs[0, 2].text(0.35, 0.35, 'Path', transform=axs[0, 2].transAxes, fontsize=config['fig_config']['font.size'],
                   color='gray',
                   verticalalignment='top')
    axs[0, 2].set_ylim(-1.2e4 / 30, 1.2e4)
    axs[0, 2].set_yticks([0, 5e3, 1e4])
    axs[0, 2].set_yticklabels(['0', '$5k$', '$10k$'])
    axs[0, 2].set_xticks([0, 433 / 3, 433 * 2 / 3, 433])
    axs[0, 2].set_xticklabels([0, '$1m$', '$2m$', '$3m$'])

    axs[1, 2].plot(sim_mc, color='#3c8990', linestyle='-.', linewidth=3)
    axs[1, 2].plot([tmp[3]] * len(sim_mc), color='#eb2f96', linestyle='-', linewidth=3)
    axs[1, 2].axvspan(spots[0], sum(spots[:2]), facecolor='g', alpha=0.10)
    axs[1, 2].axvspan(sum(spots[:3]), sum(spots[:4]), facecolor='g', alpha=0.10)
    axs[1, 2].set_ylim([-10, 300])
    axs[1, 2].set_yticks([0, 100, 200, 300])
    axs[1, 2].set_xticks([0, 433 / 3, 433 * 2 / 3, 433])
    axs[1, 2].set_xticklabels([0, '$1m$', '$2m$', '$3m$'])
    axs[1, 2].set_title(f'# Established $N=2^{{{tree_height}}}$', fontsize=config['fig_config']['font.size'])

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1),
                        fontsize=config['fig_config']['font.size'] * 0.88, labelspacing=0.5, handletextpad=0.25,
                        columnspacing=1)
    legend.get_frame().set_facecolor('#f7f7f7')
    legend.get_frame().set_edgecolor('#f7f7f7')

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(top=0.85, hspace=0.35)

    fig.savefig(config['output_dir'] + 'Fig10_real_world_case_studies.pdf')
    fig.savefig(config['output_dir'] + 'Fig10_real_world_case_studies.png')


def main():
    parser = argparse.ArgumentParser(description='Prepare the data used by Figure-11 and draw the resultant figures.')
    parser.add_argument('-d', '--draw', action='store_true', help='Draw the resultant figures.')
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

    if args.draw:
        print("# Drawing the figures ...")
        rcParams.update(config['fig_config'])

        fig, axs = plt.subplots(2, 3, figsize=(11.4, 7.1))
        draw_figs(fig, axs)
        plt.show()
        print(f"\tFigure saved to {config['output_dir'] + 'Fig10_real_world_case_studies.pdf'}\n")


if __name__ == '__main__':
    main()
