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
from math import ceil, log2, floor
from config import default_para as config

dataset_dir = config['datasets_dir']


def get_MSRC_info():
    MSRC_path = dataset_dir + 'MSRC/'
    csv_files = [f for f in os.listdir(MSRC_path) if f.endswith('.csv')]
    print(f'\t# Reading MSRC/{csv_files[0]} ...')
    all_sizes = []
    dataset_size = 0
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(MSRC_path, csv_file), usecols=[4, 5], low_memory=False)
        new_column_names = ['Offset', 'Size']
        df.columns = new_column_names

        df['Offset'] = pd.to_numeric(df['Offset'], errors='coerce')
        df = df.dropna(subset=['Offset'])

        max_offset = df['Offset'].max()
        dataset_size += max_offset
        all_sizes.extend(df['Size'].tolist())
    for i in range(len(all_sizes)):
        all_sizes[i] = int(all_sizes[i])

    print(
        f"\tTotal data volume: {dataset_size / 2 ** 30:.2f}GB, average file size: {np.mean(all_sizes) / 2 ** 10:.2f}kB")

    return dataset_size / 2 ** 30, np.mean(all_sizes) / 2 ** 10, np.quantile(all_sizes, 0.75) / 2 ** 10


def get_NIH_info():
    NIH_path = dataset_dir + 'NIH/'
    print(f'Reading NIH ...')

    all_sizes = []

    for dirpath, dirnames, filenames in os.walk(NIH_path):
        for filename in filenames:
            if filename.endswith('.png'):
                file_path = os.path.join(dirpath, filename)
                all_sizes.append(os.path.getsize(file_path))

    file_count = len(all_sizes)
    total_size = sum(all_sizes)
    mean_size = np.mean(all_sizes)
    percentile_75_size = np.percentile(all_sizes, 75)
    return file_count, total_size / 2 ** 30, mean_size / 2 ** 10, percentile_75_size / 2 ** 10


def get_COVID_info():
    COVID_path = dataset_dir + 'COVID/'
    print(f'Reading COVID ...')

    all_sizes = []

    for dirpath, dirnames, filenames in os.walk(COVID_path):
        for filename in filenames:
            if not filename.endswith('.txt'):
                file_path = os.path.join(dirpath, filename)
                all_sizes.append(os.path.getsize(file_path))

    file_count = len(all_sizes)
    total_size = sum(all_sizes)
    mean_size = np.mean(all_sizes)
    percentile_75_size = np.percentile(all_sizes, 75)
    return file_count, total_size / 2 ** 30, mean_size / 2 ** 10, percentile_75_size / 2 ** 10


def get_AliCloud_info():
    AliCloud_path = dataset_dir + 'AliCloud/io_traces_32.csv'
    all_sizes = []
    dataset_size = 0
    print(f"\t# Reading AliCloud/io_traces_32.csv ...")
    df = pd.read_csv(AliCloud_path, usecols=[2, 3], low_memory=False)
    new_column_names = ['Offset', 'Size']
    df.columns = new_column_names

    df['Offset'] = pd.to_numeric(df['Offset'], errors='coerce')
    df = df.dropna(subset=['Offset'])

    max_offset = df['Offset'].max()
    dataset_size += max_offset
    all_sizes.extend(df['Size'].tolist())
    for i in range(len(all_sizes)):
        all_sizes[i] = int(all_sizes[i])

    print(
        f"\tTotal data volume: {dataset_size / 2 ** 30:.2f}GB, average file size: {np.mean(all_sizes) / 2 ** 10:.2f}kB")

    return dataset_size / 2 ** 30, np.mean(all_sizes) / 2 ** 10, np.quantile(all_sizes, 0.75) / 2 ** 10


def get_Twitter_info():
    Twitter_path = dataset_dir + 'Twitter/cluster11.2_tripped.csv'
    all_sizes = []
    print(f"\t# Reading Twitter/cluster11.2_tripped.csv ...")
    dataset_size = 267.71 * 2 ** 30
    df = pd.read_csv(Twitter_path, usecols=[2, 3], low_memory=False)
    all_sizes.extend(df['Size'].tolist())
    for i in range(len(all_sizes)):
        all_sizes[i] = int(all_sizes[i])

    print(
        f"\tTotal data volume: {dataset_size / 2 ** 30:.2f}GB, average file size: {np.mean(all_sizes) / 2 ** 10:.2f}kB")

    return dataset_size / 2 ** 30, np.mean(all_sizes) / 2 ** 10, np.quantile(all_sizes, 0.75) / 2 ** 10


def getBlowUp(total_volume, avg_file_size, block_size, bucket_size=8, a_num=8, s_num=12, c_batch=8,
              mas_stash_size=41):
    height = ceil(log2(total_volume * 2 // block_size // bucket_size * 2))
    path_path = height * block_size * bucket_size
    ring_bucket = concur_bucket = block_size * (bucket_size + s_num)
    ring_path = concur_path = height * ring_bucket

    path_cost = path_path * 2
    ring_cost = block_size + (1 / a_num + 1 / s_num) * ring_path * 2
    concur_cost = block_size * (
            5 + 6 * c_batch + (2 + 1 / c_batch) * (c_batch + mas_stash_size) + 2 * mas_stash_size) + 1 / c_batch * (
                          concur_path * 2 + concur_bucket * log2(c_batch) + block_size * 2)

    if avg_file_size < block_size:
        path_blowup = path_cost / avg_file_size
        ring_blowup = ring_cost / avg_file_size
        concur_blowup = concur_cost / avg_file_size
    else:
        path_blowup = path_cost / block_size
        ring_blowup = ring_cost / block_size
        concur_blowup = concur_cost / block_size

    return path_blowup, ring_blowup, concur_blowup, height


def get_opt_block_size(ms_size, ms_mean):
    small_blk = 2 ** floor(log2(ms_mean)) * 2 ** 10
    larger_blk = 2 ** ceil(log2(ms_mean)) * 2 ** 10
    _, small_ring, _, height = getBlowUp(ms_size * 2 ** 30, ms_mean, small_blk)
    _, larger_ring, _, height = getBlowUp(ms_size * 2 ** 30, ms_mean, larger_blk)

    if small_ring < larger_ring:
        return small_blk / 2 ** 10, small_ring, larger_ring, height
    else:
        return larger_blk / 2 ** 10, larger_ring, small_ring, height


def get_info():
    ms_size, ms_mean, ms_p75 = get_MSRC_info()
    opt_block_size, opt_blowup, _, tree_height = get_opt_block_size(ms_size, ms_mean)
    print(
        f"\tOptimal block size: {int(opt_block_size)}kB, \tblowup: {opt_blowup / 1e3:.2f}k, \ttree height: {tree_height}\n")

    ali_size, ali_mean, ali_p75 = get_AliCloud_info()
    opt_block_size, opt_blowup, _, tree_height = get_opt_block_size(ali_size, ali_mean)
    print(
        f"\tOptimal block size: {int(opt_block_size)}kB, \tblowup: {opt_blowup / 1e3:.2f}k, \ttree height: {tree_height}\n")

    twi_size, twi_mean, twi_p75 = get_Twitter_info()
    opt_block_size, opt_blowup, _, tree_height = get_opt_block_size(twi_size, twi_mean)
    print(
        f"\tOptimal block size: {int(opt_block_size * 1024)}B, \tblowup: {opt_blowup / 1e3:.2f}k, \ttree height: {tree_height}\n")

    print(f"\t# ChestX-ray8 datasets ...")
    nih_total, nih_size, nih_mean, nih_p75 = 112120, 41.9630116764456, 392.44922343596704, 421.0810546875
    opt_block_size, opt_blowup, _, tree_height = get_opt_block_size(nih_size, nih_mean)
    print(
        f"\tTotal data volume: {nih_size:.2f}GB, average file size: {nih_mean:.2f}kB")
    print(
        f"\tOptimal block size: {int(opt_block_size)}kB, \tblowup: {opt_blowup / 1e3:.2f}k, \ttree height: {tree_height}\n")

    print(f"\t# COVIDx datasets ...")
    co_total, co_size, co_mean, co_p75 = 84819, 29.019836258143187, 358.7581063702561, 450.54638671875
    opt_block_size, opt_blowup, _, tree_height = get_opt_block_size(co_size, co_mean)
    print(
        f"\tTotal data volume: {co_size:.2f}GB, average file size: {co_mean:.2f}kB")
    print(
        f"\tOptimal block size: {int(opt_block_size)}kB, \tblowup: {opt_blowup / 1e3:.2f}k, \ttree height: {tree_height}\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare the data used by Figure-11 and draw the resultant figures.')
    parser.add_argument('-p', '--prepare', action='store_true', help='Processing the data.')
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

    if args.prepare:
        print("# Processing datasets ...")
        get_info()
        print(f"\tThe detailed code of getting ChestX-ray8 and COVIDx statistics is ommited here")


if __name__ == '__main__':
    main()
