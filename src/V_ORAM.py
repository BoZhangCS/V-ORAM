import os
import pickle

os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import argparse

from copy import deepcopy
from random import random, choice, sample, randint
from os import urandom

from src.AccessInfo import AccessInfo
from src.Path_ORAM import Path_ORAM
from src.Ring_ORAM import Ring_ORAM
from src.ConcurORAM import ConcurORAM
from src.BTree import BTree, dummy_block
from config import default_para as config


class V_ORAM():
    """
    A V-ORAM implementation that supports multiple ORAM schemes.

    Attributes:
        SUPPORTED_SIDS (list): List of supported service identifiers.
        KEY_LEN (int): Length of the encryption key.
        maxStashSize (int): Maximum size of the stash.
        height (int): Height of the ORAM tree.
        bucket_size (int): Size of each bucket.
        block_size (int): Size of each block.
        s_num (int): Number of dummy blocks in each bucket.
        a_num (int): Number of accesses in each batch.
        c_batch (int): Number of concurrent batches.
        curr_ORAM (Ring_ORAM): Current ORAM instance.
        dummy_blocks (list): List of dummy blocks.
        record_map (dict): Dictionary to record access counts for each bucket.
        curr_service (str): Current service type.
        curr_limit (int): Current limit for the number of slots in the ORAM.
        request_cache_for_concur (list): Cache for concurrent requests.
        limits (dict): Dictionary of limits for each service type.
        curr_info (AccessInfo): Current access information.
        pre_info (AccessInfo): Previous access information.

    Methods:
        __init__(height, bucket_size=8, block_size=4096, s_num=12, a_num=8, c_batch=8, maxStashSize=62):
            Initializes the V-ORAM instance with the given parameters.

        get_access_info():
            Returns the access information for the current period.

        _write_bucket(position, i):
            Writes blocks from the stash to a specific bucket.

        evictRecord(read_buckets):
            Evicts records from the read buckets.

        access(op, address, data_prime, sid):
            Performs an access operation (read or write) on the ORAM tree.

        switch_to(service_type):
            Switches the current ORAM service to the specified type.

        path_to_ring():
            Converts the current Path ORAM to a Ring ORAM.

        ring_to_path():
            Converts the current Ring ORAM to a Path ORAM.

        concur_to_ring():
            Converts the current ConcurORAM to a Ring ORAM.

        ring_to_concur():
            Converts the current Ring ORAM to a ConcurORAM.
    """
    SUPPORTED_SIDS = [
        'path',
        'ring',
        'concur'
    ]
    KEY_LEN = 32

    def __init__(self, height, bucket_size=8, block_size=4096, s_num=12, a_num=8, c_batch=8, maxStashSize=62):
        self.maxStashSize = maxStashSize
        self.height = height
        self.bucket_size = bucket_size
        self.block_size = block_size
        self.s_num = s_num
        self.a_num = a_num
        self.c_batch = c_batch

        self.curr_ORAM = Ring_ORAM(height=height, bucket_size=bucket_size, block_size=block_size, s_num=s_num,
                                   a_num=a_num)
        self.dummy_blocks = deepcopy(self.curr_ORAM.dummy_blocks)
        self.record_map = {}
        self.curr_service = 'ring'
        self.curr_limit = self.s_num  # the limit is slot num in the ring oram from the start

        self.request_cache_for_concur = []
        self.limits = {
            'path': 1,
            'ring': self.s_num,
            'concur': self.s_num,
        }

        # Data amount so far, in Bytes
        self.curr_info = AccessInfo()
        self.pre_info = deepcopy(self.curr_info)

    def get_access_info(self):
        return AccessInfo(
            down_sync=self.curr_info.down_sync - self.pre_info.down_sync,
            down_async=self.curr_info.down_async - self.pre_info.down_async,
            up_sync=self.curr_info.up_sync - self.pre_info.up_sync,
            up_async=self.curr_info.up_async - self.pre_info.up_async,
            evict_record_flag=self.curr_info.evict_record_flag,
            rtt_async=self.curr_info.rtt_async - self.pre_info.rtt_async,
            rtt_sync=self.curr_info.rtt_sync - self.pre_info.rtt_sync,
        )

    def _write_bucket(self, position, i):
        tmp_stash = {}
        min_leaf, max_leaf = BTree.get_leave_range(position, self.height)
        for address, block in self.curr_ORAM.stash.items():
            leaf_id = self.curr_ORAM.position_map[address]
            if leaf_id >= min_leaf and leaf_id <= max_leaf:
                tmp_stash[address] = deepcopy(block)

        tmp_len = min(len(tmp_stash), self.bucket_size)
        selected_address = sample(list(tmp_stash.keys()), tmp_len)

        for j in range(tmp_len):
            self.curr_ORAM.stash.pop(selected_address[j])
            block = tmp_stash[selected_address[j]]
            self.curr_ORAM[position][j] = deepcopy(block)

            self.curr_ORAM.address_map[position][j] = selected_address[j]

        for j in range(tmp_len, self.bucket_size):
            self.curr_ORAM.address_map[position][j] = -1
            self.curr_ORAM[position][j] = dummy_block(self.block_size)

        for j in range(self.bucket_size, self.bucket_size + self.s_num):
            self.curr_ORAM.address_map[position][j] = 1
            self.curr_ORAM[position][j] = deepcopy(self.curr_ORAM.dummy_blocks[i])

    def evictRecord(self, read_buckets):
        bucket_data = (self.bucket_size + self.s_num) * self.block_size
        tmp_down = 0
        tmp_up = 0
        evict_record_flag = False
        for bid in read_buckets:
            if bid not in self.record_map:
                continue
            cnt, limit = self.record_map[bid]
            if cnt >= limit:
                for i in range(self.bucket_size):
                    tmp_address = self.curr_ORAM.address_map[bid][i]
                    if tmp_address != -1:
                        self.curr_ORAM.stash[tmp_address] = self.curr_ORAM[bid][i]
                self._write_bucket(bid, BTree.get_ids(bid)[0])
                self.record_map[bid] = (0, self.curr_limit)
                for j in range(self.bucket_size, self.bucket_size + self.s_num):
                    self.curr_ORAM.address_map[bid][j] = 1
                self.curr_ORAM.count[bid] = 0

                evict_record_flag = True
                tmp_down += bucket_data
                tmp_up += bucket_data

        self.curr_info.down_sync += tmp_down
        self.curr_info.up_sync += tmp_up
        if evict_record_flag:
            self.curr_info.rtt_sync += 1
            self.curr_info.evict_record_flag += 1
        return tmp_down, tmp_up, evict_record_flag

    def access(self, op, address, data_prime, sid):
        self.curr_info.evict_record_flag = False
        self.pre_info = deepcopy(self.curr_info)

        if sid not in self.SUPPORTED_SIDS:
            raise ValueError(f'Unsupported service type: {sid}')
        if sid != self.curr_service:
            self.switch_to(sid)
        self.curr_limit = self.limits[self.curr_service]

        # normal client access
        result_data = None
        if self.curr_service != 'concur':
            if self.curr_service == 'path':
                # Path does not need EvictRecord, but needs to be marked
                result_data, read_buckets, evicted_buckets, info = self.curr_ORAM.access(op, address, data_prime)
                for bid in read_buckets:
                    self.record_map[bid] = (1, self.curr_limit)
            else:
                # EvictRecord
                # Pre-evict the bucket to be read from Path ORAM
                if address in self.curr_ORAM.stash or self.curr_ORAM.position_map[address] == -1:
                    target_leaf = randint(0, self.curr_ORAM.leaf_num - 1)
                else:
                    target_leaf = self.curr_ORAM.position_map[address]
                read_buckets = []
                for i in range(self.height):
                    position = self.curr_ORAM.get_position(target_leaf, i)
                    read_buckets.append(position)

                self.evictRecord(read_buckets)

                result_data, read_buckets, evicted_buckets, info = self.curr_ORAM.access(op, address, data_prime)

                for bid in read_buckets:
                    if bid not in self.record_map.keys():
                        self.record_map[bid] = (1, self.curr_limit)
                    else:
                        cnt, limit = self.record_map[bid]
                        self.record_map[bid] = (cnt + 1, min(self.curr_limit, limit))

                for bid in evicted_buckets:
                    self.record_map[bid] = (0, self.curr_limit)

                self.evictRecord(read_buckets)

            self.curr_info.down_sync += info.down_sync
            self.curr_info.up_sync += info.up_sync
            self.curr_info.rtt_sync += info.rtt_sync
        else:
            self.request_cache_for_concur.append((op, address, data_prime))
            if len(self.request_cache_for_concur) == self.c_batch:
                assert isinstance(self.curr_ORAM, ConcurORAM)
                self.curr_ORAM.add_record_map(self.record_map)
                result_data, info = self.curr_ORAM.access(self.request_cache_for_concur)
                self.request_cache_for_concur = []

                self.curr_info.down_sync += info.down_sync
                self.curr_info.down_async += info.down_async
                self.curr_info.up_sync += info.up_sync
                self.curr_info.up_async += info.up_async
                self.curr_info.rtt_sync += info.rtt_sync
                self.curr_info.rtt_async += info.rtt_async
                if info.evict_record_flag:
                    self.curr_info.evict_record_flag = True

        return result_data, self.get_access_info()

    def switch_to(self, service_type):
        # switch_back_to_base_ORAM
        if self.curr_service == 'path':
            pass
            self.path_to_ring()
        elif self.curr_service == 'ring':
            pass
        elif self.curr_service == 'concur':
            self.concur_to_ring()

        # switch_to_target_ORAM
        if service_type == 'path':
            self.ring_to_path()
        elif service_type == 'ring':
            pass
        elif service_type == 'concur':
            self.ring_to_concur()

        self.curr_service = service_type

    def path_to_ring(self):
        tmp_ORAM = Ring_ORAM(height=self.height, bucket_size=self.bucket_size, block_size=self.block_size,
                             s_num=self.s_num, a_num=self.a_num)
        tmp_ORAM.layers = self.curr_ORAM.layers
        tmp_ORAM.stash = self.curr_ORAM.stash
        tmp_ORAM.address_map = self.curr_ORAM.address_map
        tmp_ORAM.position_map = self.curr_ORAM.position_map
        self.curr_ORAM = tmp_ORAM

    def ring_to_path(self):
        tmp_ORAM = Path_ORAM(height=self.height, bucket_size=self.bucket_size, block_size=self.block_size)
        tmp_ORAM.layers = self.curr_ORAM.layers
        tmp_ORAM.stash = self.curr_ORAM.stash
        tmp_ORAM.address_map = self.curr_ORAM.address_map
        tmp_ORAM.position_map = self.curr_ORAM.position_map
        self.curr_ORAM = tmp_ORAM

    def concur_to_ring(self):
        tmp_stash = {}
        if isinstance(self.curr_ORAM, ConcurORAM):
            # Commit the left requests
            curr_len = len(self.curr_ORAM.evictionLog) - self.curr_ORAM.evictionLog.count(-1)
            self.curr_ORAM.evict_batch_commit()

            # Retrieve stash from DRL and tempStash
            position_map = self.curr_ORAM.position_map
            for i in range(curr_len - 1, -1, -1):
                for add, block in self.curr_ORAM.DR_LogSet[i]:
                    if add in position_map.keys() and position_map[add] != -1 and add not in tmp_stash.keys():
                        tmp_stash[add] = block

            curr_tempStash = self.curr_ORAM.StashSet[(curr_len - 1) % self.c_batch]
            for j in range(self.maxStashSize):
                add, block = curr_tempStash[j]
                if add < len(position_map) and position_map[add] != -1:
                    tmp_stash[add] = block

        tmp_ORAM = Ring_ORAM(self.height, self.bucket_size, self.block_size, self.s_num, self.a_num)
        tmp_ORAM.layers = self.curr_ORAM.layers
        tmp_ORAM.stash = tmp_stash
        tmp_ORAM.position_map = self.curr_ORAM.position_map
        tmp_ORAM.address_map = self.curr_ORAM.address_map
        self.curr_ORAM = tmp_ORAM

    def ring_to_concur(self):
        tmp_ORAM = ConcurORAM(height=self.height, bucket_size=self.bucket_size, block_size=self.block_size,
                              c_batch=self.a_num, a_num=self.a_num,
                              s_num=self.s_num, maxStashSize=self.maxStashSize, no_map=True)
        tmp_ORAM.layers = self.curr_ORAM.layers
        # Store the stash to tempStash[c_batch-1]
        tmp_stash = []
        for add, block in self.curr_ORAM.stash.items():
            tmp_stash.append((add, block))
        if len(tmp_stash) > self.maxStashSize:
            raise Exception(f'Overflow in stash size during switching to concur')
        while len(tmp_stash) < self.maxStashSize:
            tmp_stash.append((urandom(16).hex(), urandom(self.block_size)))
        tmp_ORAM.StashSet[self.a_num - 1] = tmp_stash
        # Transform position map, address map
        tmp_ORAM.position_map = self.curr_ORAM.position_map
        tmp_ORAM.address_map = self.curr_ORAM.address_map
        self.curr_ORAM = tmp_ORAM


def main():
    parser = argparse.ArgumentParser(description='Initialize a prototype of V-ORAM.')
    parser.add_argument('-i', '--init', action='store_true', help='Initialize a V-ORAM according to config.py.')
    parser.add_argument('-w', '--write', nargs=2, metavar=('address', 'data'), help='Write the data, followed by address and data value.')
    parser.add_argument('-r', '--read', type=int, metavar='address', help='Write the data, followed by address and data value.')
    parser.add_argument('-o', '--oram', type=str, help='Execute the operation by given ORAM.')
    args = parser.parse_args()

    if args.init:
        print('Initializing V-ORAM ...')
        maxStashSize = config.get('maxStashSize')
        block_size = config.get('block_size')
        bucket_size = config.get('bucket_size')
        s_num = config.get('s_num')
        a_num = config.get('a_num')
        c_batch = config.get('c_batch')
        height = config.get('V_ORAM_height')
        v_oram = V_ORAM(height=height, bucket_size=bucket_size, block_size=block_size, s_num=s_num, a_num=a_num,
                        c_batch=c_batch, maxStashSize=maxStashSize)
        print(f'\tV-ORAM initialized with height = {height}.')

        print(f'Saving V-ORAM to v_oram.pkl ...')
        with open('v_oram.pkl', 'wb') as f:
            pickle.dump(v_oram, f)
        print('\tV-ORAM saved successfully.')

    if os.path.exists('v_oram.pkl'):
        with open('v_oram.pkl', 'rb') as f:
            v_oram = pickle.load(f)
    else:
        print(f'v_oram.pkl not found. Please initialize V-ORAM using the --init option.')
        return

    if args.oram:
        target_service = args.oram
        print(f'ORAM service changes to {target_service}.')
    else:
        target_service = v_oram.curr_service

    if args.write:
        address = int(args.write[0])
        data = args.write[1].encode('utf-8')
        # Processing data here, change here if you want to customize data type.
        if len(data) > v_oram.block_size:
            data = data[:v_oram.block_size]
        else:
            data = data.ljust(v_oram.block_size, b'\0')
        v_oram.access('write', address, data, target_service)


    if args.read:
        address = int(args.read)
        data, _ = v_oram.access('read', address, urandom(v_oram.block_size), target_service)
        data = data.rstrip(b'\0')
        print(f"Data at address {address}: {data}")

    if not args.init:
        with open('v_oram.pkl', 'wb') as f:
            pickle.dump(v_oram, f)


if __name__ == '__main__':
    main()
