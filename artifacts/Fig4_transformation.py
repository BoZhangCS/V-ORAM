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
from math import log
from tqdm import tqdm
from os import urandom
from matplotlib import pyplot as plt, rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from src.utils import Encoder, output_dir
from config import default_para as config


def simTrans(previous, target, **config):
    maxStashSize = config.get('maxStashSize')
    block_size = config.get('block_size')
    down_bw = config.get('down_bw')
    up_bw = config.get('up_bw')
    c_batch = config.get('c_batch')
    encoder = Encoder()
    if previous == 'path' and target == 'concur':
        tmp1 = simTrans('path', 'ring', **config)
        tmp2 = simTrans('ring', 'concur', **config)
        return tmp1[0] + tmp2[0], tmp1[1] + tmp2[1]
    elif previous == 'concur' and target == 'path':
        tmp1 = simTrans('concur', 'ring', **config)
        tmp2 = simTrans('ring', 'path', **config)
        return tmp1[0] + tmp2[0], tmp1[1] + tmp2[1]
    elif previous == 'concur' and target == 'ring':
        time_ = 0.1
        comm = 0
        # Communication
        # temp stash
        comm += (maxStashSize + c_batch) * block_size
        # DRL
        comm += (2 * c_batch) * c_batch * block_size
        decrypted_blocks = (maxStashSize + c_batch) + (2 * c_batch * c_batch)
        tmp_cipher = encoder.Enc(urandom(block_size), 'path')
        start_time = time.time()
        for _ in range(decrypted_blocks):
            encoder.Dec(tmp_cipher)
        time_ += (time.time() - start_time) / 1000 + comm / down_bw
    elif previous == 'ring' and target == 'concur':
        time_ = 0.1
        comm = 0
        # Communication
        # StashSet
        comm += (maxStashSize + c_batch) * block_size
        encrypted_blocks = (maxStashSize + c_batch)
        start_time = time.time()
        for _ in range(encrypted_blocks):
            encoder.Enc(urandom(block_size), 'path')
        time_ += (time.time() - start_time) / 1000 + comm / up_bw
    elif previous == 'ring' and target == 'path':
        time_ = 0.1
        comm = 4096
    elif previous == 'path' and target == 'ring':
        time_ = 0.1
        comm = 4096
    else:
        raise Exception("Invalid ORAM type")
    return time_, comm / 2 ** 20


def simDirect(previous, target, height, **config):
    # Here omit the metadata size, as the entire ORAM is large
    # The encryption is estimated through processing speed
    bucket_size = config.get('bucket_size')
    s_num = config.get('s_num')
    block_size = config.get('block_size')
    down_bw = config.get('down_bw')
    up_bw = config.get('up_bw')
    encoder = Encoder()
    tmp_cipher = encoder.Enc(urandom(block_size), 'path')

    # The actual valid storage volume, half of the data tree
    valid_block_num = ((2 ** height - 1) * bucket_size) // 2
    time_ = 0
    down_comm = 0
    up_comm = 0

    # Encryption speed
    start_time = time.time()
    for _ in range(2 ** 9):
        encoder.Dec(tmp_cipher)
    dec_mbps = block_size / 2 ** 10 / (time.time() - start_time)
    start_time = time.time()
    for _ in range(2 ** 9):
        encoder.Enc(urandom(block_size), 'path')
    enc_mbps = block_size / 2 ** 10 / (time.time() - start_time)

    # download and decrypt
    if previous == 'path':
        down_comm += (2 ** height - 1) * bucket_size * block_size
    else:
        down_comm += (2 ** height - 1) * (bucket_size + s_num) * block_size
    time_ += valid_block_num * block_size / 2 ** 20 / dec_mbps + down_comm / down_bw

    # upload and encrypt
    if target == 'path':
        up_comm += (2 ** height - 1) * bucket_size * block_size
    else:
        up_comm += (2 ** height - 1) * (bucket_size + s_num) * block_size
    time_ += valid_block_num * block_size / 2 ** 20 / enc_mbps + up_comm / up_bw
    return time_, (down_comm + up_comm) / 2 ** 20


def simMulti(height, **config):
    # Maintaining multiple instances of ORAMs, equivalent to access all ORAMs
    bucket_size = config.get('bucket_size')
    a_num = config.get('a_num')
    s_num = config.get('s_num')
    c_batch = config.get('c_batch')
    block_size = config.get('block_size')
    maxStashSize = config.get('maxStashSize')
    down_bw = config.get('down_bw')
    up_bw = config.get('up_bw')
    encoder = Encoder()
    tmp_cipher = encoder.Enc(urandom(block_size), 'path')
    time_ = 0
    down_comm = 0
    up_comm = 0

    # Encryption speed
    start_time = time.time()
    for _ in range(2 ** 10):
        encoder.Dec(tmp_cipher)
    dec_mbps = block_size / 2 ** 10 / (time.time() - start_time)
    start_time = time.time()
    for _ in range(2 ** 10):
        encoder.Enc(urandom(block_size), 'path')
    enc_mbps = block_size / 2 ** 10 / (time.time() - start_time)

    # Access time of three ORAMs, ref to Table 4 in the paper
    # Path ORAM access
    c_down = height * bucket_size * block_size
    c_up = height * bucket_size * block_size
    down_comm += c_down
    up_comm += c_up
    time_ += c_down / 2 ** 20 / dec_mbps + c_up / 2 ** 20 / enc_mbps + c_down / down_bw + c_up / up_bw

    # Ring ORAM access
    c_down = block_size + (1 / a_num + 1 / s_num) * height * (bucket_size + s_num) * block_size
    c_up = (1 / a_num + 1 / s_num) * height * (bucket_size + s_num) * block_size
    down_comm += c_down
    up_comm += c_up
    # XOR computation time is omitted here
    time_ += c_down / 2 ** 20 / dec_mbps + c_up / 2 ** 20 / enc_mbps + c_down / down_bw + c_up / up_bw

    # ConcurORAM access
    # time of DP-ORAM is omitted here
    c_down = block_size * (3 + 2 * c_batch + (2 + 1 / c_batch) * (c_batch + maxStashSize)) + (1 / c_batch) * (
            height * (bucket_size + s_num) + bucket_size * log(c_batch, 2))
    c_up = (2 + 4 * c_batch + 2 * maxStashSize) * block_size + (1 / c_batch) * (height * (bucket_size + s_num))
    down_comm += c_down
    up_comm += c_up
    time_ += c_down / 2 ** 20 / dec_mbps + c_up / 2 ** 20 / enc_mbps + c_down / down_bw + c_up / up_bw
    return time_, (down_comm + up_comm) / 2 ** 20


def simRing(height, **config):
    # The simulation of Ring ORAM, ref to Table 4 in the paper
    bucket_size = config.get('bucket_size')
    s_num = config.get('s_num')
    block_size = config.get('block_size')
    down_bw = config.get('down_bw')
    up_bw = config.get('up_bw')
    encoder = Encoder()
    tmp_cipher = encoder.Enc(urandom(block_size), 'path')
    time_ = 0
    down_comm = 0
    up_comm = 0
    # Encryption speed
    start_time = time.time()
    for _ in range(2 ** 10):
        encoder.Dec(tmp_cipher)
    dec_mbps = block_size / 2 ** 10 / (time.time() - start_time)
    start_time = time.time()
    for _ in range(2 ** 10):
        encoder.Enc(urandom(block_size), 'path')
    enc_mbps = block_size / 2 ** 10 / (time.time() - start_time)

    # Access time of Ring ORAM, ref to Table 4 in the paper
    c_down = block_size + (1 / 8 + 1 / s_num) * height * (bucket_size + s_num) * block_size
    c_up = (1 / 8 + 1 / s_num) * height * (bucket_size + s_num) * block_size
    down_comm += c_down
    up_comm += c_up
    # XOR computation time is omitted here
    time_ += c_down / 2 ** 20 / dec_mbps + c_up / 2 ** 20 / enc_mbps + c_down / down_bw + c_up / up_bw
    return time_, (down_comm + up_comm) / 2 ** 20


def get_price(time_cost, comm):
    # AWS
    # 0.544 $/hr for 16 vCPU instance c6g.4xlarge
    # 0.01  $/GB
    # AliCloud
    # 0.491 $/hr for 16 vCPU instance ecs.u1-c1m1.4xlarge
    # 0.076 $/GB
    return time_cost / 3600 * 0.544 + comm / 1024 * 0.01


def prepare_csv(info, start_height, end_height):
    columns = [
        'range', 'r2c_Dt', 'c2r_Dt', 'r2p_Dt', 'p2r_Dt', 'c2p_Dt', 'p2c_Dt',
        'c2p_Tt', 'p2c_Tt', 'r2c_Tt', 'c2r_Tt', 'r2p_Tt', 'p2r_Tt', 'multi_t', 'ring_t',
        'r2c_Dc', 'c2r_Dc', 'r2p_Dc', 'p2r_Dc', 'c2p_Dc', 'p2c_Dc',
        'r2c_Tc', 'c2r_Tc', 'r2p_Tc', 'p2r_Tc', 'c2p_Tc', 'p2c_Tc', 'multi_c', 'ring_c',
        'rp_Dm', 'rp_Tm', 'rc_Dm', 'rc_Tm', 'pc_Dm', 'pc_Tm', 'multi_m', 'ring_m'
    ]
    df = pd.DataFrame({col: [0] * (end_height - start_height + 1) for col in columns})
    end_height += 1
    with tqdm(total=end_height - start_height, ncols=80) as pbar:
        for curr_height in range(start_height, end_height):
            pbar.set_description(f'\tCurrent height: {curr_height}')
            i = curr_height - start_height
            df.loc[i, 'range'] = 2 ** curr_height
            df.loc[i, 'r2c_Dt'] = simDirect('ring', 'concur', height=curr_height, **info)[0]
            df.loc[i, 'c2r_Dt'] = simDirect('concur', 'ring', height=curr_height, **info)[0]
            df.loc[i, 'r2p_Dt'] = simDirect('ring', 'path', height=curr_height, **info)[0]
            df.loc[i, 'p2r_Dt'] = simDirect('path', 'ring', height=curr_height, **info)[0]
            df.loc[i, 'c2p_Dt'] = simDirect('concur', 'path', height=curr_height, **info)[0]
            df.loc[i, 'p2c_Dt'] = simDirect('path', 'concur', height=curr_height, **info)[0]
            df.loc[i, 'c2p_Tt'] = simTrans('concur', 'path', **info)[0]
            df.loc[i, 'p2c_Tt'] = simTrans('path', 'concur', **info)[0]
            df.loc[i, 'r2c_Tt'] = simTrans('ring', 'concur', **info)[0]
            df.loc[i, 'c2r_Tt'] = simTrans('concur', 'ring', **info)[0]
            df.loc[i, 'r2p_Tt'] = simTrans('ring', 'path', **info)[0]
            df.loc[i, 'p2r_Tt'] = simTrans('path', 'ring', **info)[0]
            df.loc[i, 'multi_t'] = simMulti(height=curr_height, **info)[0]
            df.loc[i, 'ring_t'] = simRing(height=curr_height, **info)[0]
            df.loc[i, 'r2c_Dc'] = simDirect('ring', 'concur', height=curr_height, **info)[1]
            df.loc[i, 'c2r_Dc'] = simDirect('concur', 'ring', height=curr_height, **info)[1]
            df.loc[i, 'r2p_Dc'] = simDirect('ring', 'path', height=curr_height, **info)[1]
            df.loc[i, 'p2r_Dc'] = simDirect('path', 'ring', height=curr_height, **info)[1]
            df.loc[i, 'c2p_Dc'] = simDirect('concur', 'path', height=curr_height, **info)[1]
            df.loc[i, 'p2c_Dc'] = simDirect('path', 'concur', height=curr_height, **info)[1]
            df.loc[i, 'r2c_Tc'] = simTrans('ring', 'concur', **info)[1]
            df.loc[i, 'c2r_Tc'] = simTrans('concur', 'ring', **info)[1]
            df.loc[i, 'r2p_Tc'] = simTrans('ring', 'path', **info)[1]
            df.loc[i, 'p2r_Tc'] = simTrans('path', 'ring', **info)[1]
            df.loc[i, 'c2p_Tc'] = simTrans('concur', 'path', **info)[1]
            df.loc[i, 'p2c_Tc'] = simTrans('path', 'concur', **info)[1]
            df.loc[i, 'multi_c'] = simMulti(height=curr_height, **info)[1]
            df.loc[i, 'ring_c'] = simRing(height=curr_height, **info)[1]
            pbar.update(1)

    df['rp_Dm'] = get_price((df['r2p_Dt'] + df['p2r_Dt']) / 2, (df['r2p_Dc'] + df['p2r_Dc']) / 2)
    df['rp_Tm'] = get_price((df['r2p_Tt'] + df['p2r_Tt']) / 2, (df['r2p_Tc'] + df['p2r_Tc']) / 2)
    df['rc_Dm'] = get_price((df['r2c_Dt'] + df['c2r_Dt']) / 2, (df['r2c_Dc'] + df['c2r_Dc']) / 2)
    df['rc_Tm'] = get_price((df['r2c_Tt'] + df['c2r_Tt']) / 2, (df['r2c_Tc'] + df['c2r_Tc']) / 2)
    df['pc_Dm'] = get_price((df['c2p_Dt'] + df['p2c_Dt']) / 2, (df['c2p_Dc'] + df['p2c_Dc']) / 2)
    df['pc_Tm'] = get_price((df['c2p_Tt'] + df['p2c_Tt']) / 2, (df['c2p_Tc'] + df['p2c_Tc']) / 2)
    df['multi_m'] = get_price(df['multi_t'], df['multi_c'])
    df['ring_m'] = get_price(df['ring_t'], df['ring_c'])

    df.to_csv(config['output_dir'] + 'trans_time_comm.csv', index=False)


def draw_figs(fig, axs, csv_path, ignore_legend=False, ignore_summary=False, saveFig=True):
    df = pd.read_csv(csv_path)
    for ax in axs:
        ax.tick_params(axis='x', which='major', pad=5)
        ax.grid(axis='y', color='#e6e6e6', linewidth=2, )
        ax.grid(axis='x', color='#e6e6e6', linewidth=2, )
        ax.set_yscale('log', base=10)
        ax.set_xscale('log', base=2)
        ax.set_xticks([2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22, 2 ** 24])
        ax.set_xlabel('Buckets')

    axs[0].plot(df['range'], (df['r2c_Dt'] + df['c2r_Dt']) / 2, ':s', linewidth=3, color='#1677ff', markersize=10,
                label='(D) Ring$\\Leftrightarrow$Concur')
    axs[0].plot(df['range'], (df['c2r_Tt'] + df['r2c_Tt']) / 2, '-s', linewidth=3, color='#1677ff', markersize=10,
                label='(T) Ring$\\Leftrightarrow$Concur')
    axs[0].plot(df['range'], (df['p2c_Dt'] + df['c2p_Dt']) / 2, ':o', linewidth=3, color='#52c41a', markersize=10,
                label='(D) Path$\\Leftrightarrow$Concur')
    axs[0].plot(df['range'], (df['p2c_Tt'] + df['c2p_Tt']) / 2, '-o', linewidth=3, color='#52c41a', markersize=10,
                label='(T) Path$\\Leftrightarrow$Concur')
    axs[0].plot(df['range'], df['multi_t'], '-D', linewidth=3, color='#fa8c16', markersize=10, label='(M) All')
    axs[0].set_yticks([1e-1, 1e1, 1e3, 1e5])
    axs[0].set_ylim(0.2e-1, 1.1e6)
    axs[0].set_ylabel('Processing time (s)', labelpad=0)

    axins = inset_axes(axs[0], width="40%", height="40%", bbox_to_anchor=(0.3, -0.03, 0.95, 0.9),
                       bbox_transform=axs[0].transAxes, loc='center')
    axins.plot(df['range'], (df['p2c_Tt'] + df['c2p_Tt']) / 2, '-o', linewidth=3, color='#52c41a', markersize=10)
    axins.plot(df['range'], (df['c2r_Tt'] + df['r2c_Tt']) / 2, '-s', linewidth=3, color='#1677ff', markersize=10)
    axins.plot(df['range'], df['multi_t'], '-D', linewidth=3, color='#fa8c16', markersize=10)
    axins.set_xlim(2 ** 14, 2 ** 20)
    axins.set_ylim(1.5e-1, 4e-1)
    axins.yaxis.set_visible(False)
    axins.xaxis.set_visible(False)
    axins.set_xscale('log', base=2)
    axins.set_yscale('log', base=10)
    mark_inset(axs[0], axins, loc1=2, loc2=4, fc="none", ec="0", zorder=10)

    axs[1].plot(df['range'], (df['r2c_Dc'] + df['c2r_Dc']) / 2, ':s', linewidth=3, color='#1677ff', markersize=10)
    axs[1].plot(df['range'], (df['c2r_Tc'] + df['r2c_Tc']) / 2, '-s', linewidth=3, color='#1677ff', markersize=10)
    axs[1].plot(df['range'], (df['p2c_Dc'] + df['c2p_Dc']) / 2, ':o', linewidth=3, color='#52c41a', markersize=10)
    axs[1].plot(df['range'], (df['p2c_Tc'] + df['c2p_Tc']) / 2, '-o', linewidth=3, color='#52c41a', markersize=10)
    axs[1].plot(df['range'], df['multi_c'], '-D', linewidth=3, color='#fa8c16', markersize=10)
    axs[1].set_yticks([1e-3, 1e0, 1e3, 1e6])
    axs[1].set_ylim(0.4e-1, 1.1e7)
    axs[1].set_ylabel('Comm. cost (MB)', labelpad=0)

    axs[2].plot(df['range'], df['rc_Dm'], ':s', linewidth=3, color='#1677ff', markersize=10)
    axs[2].plot(df['range'], df['rc_Tm'], '-s', linewidth=3, color='#1677ff', markersize=10)
    axs[2].plot(df['range'], df['pc_Dm'], ':o', linewidth=3, color='#52c41a', markersize=10)
    axs[2].plot(df['range'], df['pc_Tm'], '-o', linewidth=3, color='#52c41a', markersize=10)
    axs[2].plot(df['range'], df['multi_m'], '-D', linewidth=3, color='#fa8c16', markersize=10)
    axs[2].set_yticks([1e-5, 1e-3, 1e-1, 1e1])
    axs[2].set_ylim(0.5e-5, 4e2)
    axs[2].set_ylabel('Monetary cost ($)', labelpad=0)

    if not ignore_legend:
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        legend = fig.legend(lines, labels, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.01),
                            fontsize=config['fig_config']['font.size'] * 0.88, labelspacing=0.5, handletextpad=0.25,
                            columnspacing=1)
        legend.get_frame().set_facecolor('#f7f7f7')
        legend.get_frame().set_edgecolor('#f7f7f7')

    plt.tight_layout(pad=0.2)

    if not ignore_legend:
        plt.subplots_adjust(top=0.84)
    else:
        plt.subplots_adjust(top=0.91, hspace=0.75, bottom=0.2)
    if saveFig:
        fig.savefig(output_dir + 'Fig4_transformation.pdf')

    multi_vs_ring_time = df['multi_t'] / df['ring_t']
    multi_vs_ring_comm = df['multi_c'] / df['ring_c']
    multi_vs_ring_cost = df['multi_m'] / df['ring_m']
    rc_Dt = (df["c2r_Dt"].iloc[6] + df["r2c_Dt"].iloc[6]) / 2
    rc_Dc = (df["c2r_Dc"].iloc[6] + df["r2c_Dc"].iloc[6]) / 2
    rc_Tt = (df["c2r_Tt"].iloc[6] + df["r2c_Tt"].iloc[6]) / 2
    rc_Tc = (df["c2r_Tc"].iloc[6] + df["r2c_Tc"].iloc[6]) / 2

    if (not ignore_legend) and (not ignore_summary):
        print(f'Brief summary:\n'
              f'\tCosts of direct download of Concur <-> Ring:\n'
              f'\t\tTime: {rc_Dt:.2f} s, Comm: {rc_Dc / 1024:.2f} GB, Money: {df["rc_Dm"].iloc[6]:.2f} $\n'
              f'\tCosts of transformation of Concur <-> Ring:\n'
              f'\t\tTime: {rc_Tt:.2f} s, Comm: {rc_Tc * 1024:.2f} KB, Money: {df["rc_Tm"].iloc[6] * 1e5:.2f}e-5$\n'
              f'\tSaving factors:\n'
              f'\t\tTime: {log(df["c2r_Dt"].iloc[6] / df["c2r_Tt"].iloc[6]) / log(10):.2f} X, '
              f'Comm: {log(df["c2r_Dc"].iloc[6] / df["c2r_Tc"].iloc[6]) / log(10):.2f} X, '
              f'Money: {log(df["rc_Dm"].iloc[6] / df["rc_Tm"].iloc[6]) / log(10):.2f} X\n'
              f'\tMulti vs Ring (Time): {multi_vs_ring_time.mean():.2f} X\n'
              f'\tMulti vs Ring (Comm): {multi_vs_ring_comm.mean():.2f} X\n'
              f'\tMulti vs Ring (Cost): {multi_vs_ring_cost.mean():.2f} X, (5.36 X is used in the paper)\n'
              )


def main():
    parser = argparse.ArgumentParser(description='Prepare the data used by Figure-11 and draw the resultant figures.')
    parser.add_argument('-p', '--prepare', action='store_true', help='Prepare the data.')
    parser.add_argument('-d', '--draw', action='store_true', help='Draw the resultant figures.')
    parser.add_argument('-c', '--compare', action='store_true',
                        help='Generate comparison with the graphs in the paper.')
    parser.add_argument('-s', '--start', type=int, default=10, help='Start height (default: 10)')
    parser.add_argument('-e', '--end', type=int, default=24, help='End height (default: 24)')
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

    if args.prepare:
        print("# Preparing the data ...")
        start_height = args.start
        end_height = args.end
        prepare_csv(config, start_height, end_height)
        print(f"\tData saved to {config['output_dir'] + 'trans_time_comm.csv'}\n")

    if args.draw:
        print("# Drawing the figures ...")
        rcParams.update(config['fig_config'])

        fig, axs = plt.subplots(1, 3, figsize=(17.16, 3.96))
        draw_figs(fig, axs, config['output_dir'] + 'trans_time_comm.csv')
        print(f"\tFigure saved to {config['output_dir'] + 'Fig4_transformation.pdf'}\n")

        if not args.compare:
            plt.show()

    if args.compare:
        print("# Comparing the results with the graphs in the paper ...")
        rcParams.update(config['fig_config'])
        fig, axs = plt.subplots(2, 3, figsize=(17.16, 7.7))
        draw_figs(fig, axs[0], config['output_dir'] + 'trans_time_comm.csv', ignore_summary=True, saveFig=False)
        draw_figs(fig, axs[1], config['paper_data_dir'] + 'trans_time_comm.csv', ignore_legend=True, saveFig=False)
        fig.text(0.5, 0.5, 'Artifact Evaluation', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.text(0.5, 0.05, 'Paper Figure 4', ha='center', va='center',
                 fontsize=config['fig_config']['font.size'] * 1.5)
        fig.savefig(config['output_dir'] + 'Fig4_transformation_compare.png')
        print(f"\tFigure saved to {config['output_dir'] + 'Fig4_transformation_compare.png'}\n")
        print("\tThe first row is the figures generated by the artifact.\n"
              "\tThe second row is the figures in the paper.")
        plt.show()


if __name__ == '__main__':
    main()
