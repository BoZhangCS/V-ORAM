import copy
import os.path
import pickle
from math import floor, ceil
from os import urandom
from config import default_para as config

BLOCK_SIZE = config['block_size']
DIRECT_NAME = config['tree_data_dir']
DATA_FILE_SIZE = config['data_file_size']
BUCKETS_PER_FILE = floor(DATA_FILE_SIZE / BLOCK_SIZE)

# Using same dummies for runtime saving
global_dummies = {
    BLOCK_SIZE: urandom(BLOCK_SIZE),
}


def dummy_block(block_size=4096):
    if block_size in global_dummies.keys():
        return global_dummies[block_size]
    else:
        global_dummies[block_size] = urandom(block_size)
        return urandom(block_size)


def _data_str(data):
    if type(data) == bytes:
        return data[:4].hex()
    else:
        return data


# Buckets are stored in files in /data/tree_data/
class Bucket:
    def __init__(self, size=None, block_size=4096, bucket_id=None):
        self.size = size if size else 4
        self.bucket_id = bucket_id
        self.block_size = block_size
        self.data_file_size = DATA_FILE_SIZE
        self.bucket_per_file = self.data_file_size // self.block_size // self.size
        self.file_name, self.offset = self._get_file_loc()

    def _get_file_loc(self):
        file_index = self.bucket_id // self.bucket_per_file
        offset = self.bucket_id % self.bucket_per_file
        file_name = f'{DIRECT_NAME}tree_data_{file_index}.bin'
        return file_name, offset

    def __getitem__(self, bios):
        with open(self.file_name, 'rb') as f:
            f.seek((self.offset * self.size) * self.block_size)
            tmp = []
            for _ in range(self.size):
                tmp.append(f.read(self.block_size))
            return tmp[bios]

    def __setitem__(self, bios, block):
        with open(self.file_name, 'r+b') as f:
            f.seek((self.offset * self.size + bios) * self.block_size)
            f.write(block)


# Binary tree
class BTree:
    def __init__(self, height, address_size=None, block_size=None, bucket_size=None):
        self.height = height
        self.address_size = address_size if address_size else 32  # in bits
        self.block_size = block_size if block_size else 4 * 1024  # in bytes
        self.bucket_size = bucket_size if bucket_size else 4  # in blocks
        self.layers = []
        self.init()

    def init(self):
        total_size = 2 ** self.height * self.bucket_size * self.block_size
        total_files = ceil((total_size + 0.0) / DATA_FILE_SIZE)

        for file_index in range(total_files):
            file_name = f'{DIRECT_NAME}tree_data_{file_index}.bin'
            if not os.path.exists(file_name):
                with open(file_name, 'wb') as f:
                    f.seek(DATA_FILE_SIZE - 1)
                    f.write(b'\0')

        for i in range(self.height):
            tmp = []
            for j in range(2 ** i):
                tmp.append(
                    Bucket(size=self.bucket_size, bucket_id=2 ** i - 1 + j, block_size=self.block_size))
            self.layers.append(tmp)

    def print(self):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            for j in range(len(layer)):
                bucket = layer[j]
                for block in bucket:
                    print(f'{_data_str(block)}', end=', ')
                print('||', end=' ')
            print('\n')
        print()

    def read_path(self, leaf_ind):
        path = []
        layer_ind = self.height - 1
        position = 2 ** layer_ind + leaf_ind - 1
        while position:
            tmp = self[position]
            path.append(copy.deepcopy(tmp))
            position = (position - 1) // 2
        path.append(copy.deepcopy(self[0]))
        return path

    @classmethod
    def get_ids(self, position):
        layer_ind = 0
        tmp = position
        while tmp:
            tmp = (tmp - 1) // 2
            layer_ind += 1
        layer_bios = position - 2 ** (layer_ind) + 1
        if layer_ind == -1 or layer_bios == -1:
            raise Exception("Invalid position")
        return layer_ind, layer_bios

    def get_position(self, leaf, layer):
        # layer是从根节点向下数的层数
        leaf_position = 2 ** (self.height - 1) + leaf - 1
        tmp = self.height - 1 - layer
        while tmp:
            leaf_position = (leaf_position - 1) // 2
            tmp -= 1
        return leaf_position

    @classmethod
    def get_leave_range(self, position, height):
        layer_idx, layer_bios = self.get_ids(position)
        min_leaf = position
        max_leaf = position
        while layer_idx < height - 1:
            min_leaf = min_leaf * 2 + 1
            max_leaf = max_leaf * 2 + 2
            layer_idx += 1
        min_leaf = min_leaf - 2 ** (height - 1) + 1
        max_leaf = max_leaf - 2 ** (height - 1) + 1
        return min_leaf, max_leaf

    def g_to_l(self, big_g):
        tmp_l = big_g % (2 ** (self.height - 1))
        l = 0
        for i in range(self.height - 1):
            if tmp_l % 2 == 1:
                l = l * 2 + 2
            else:
                l = l * 2 + 1
            tmp_l = tmp_l // 2
        l = l - 2 ** (self.height - 1) + 1
        return l

    def __getitem__(self, position) -> Bucket:
        layer_ind, layer_bios = self.get_ids(position)
        return self.layers[layer_ind][layer_bios]

    def __setitem__(self, position, bucket):
        layer_ind, layer_bios = self.get_ids(position)
        self.layers[layer_ind][layer_bios] = bucket


if __name__ == '__main__':
    pass