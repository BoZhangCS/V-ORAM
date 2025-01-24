from copy import deepcopy
import random
from math import ceil, log2
from typing import List
from os import urandom

from src.AccessInfo import AccessInfo
from src.BTree import BTree, dummy_block, _data_str


class ConcurORAM(BTree):

    def __init__(self, height, bucket_size=8, block_size=4096, c_batch=4, maxStashSize=62, a_num=8, s_num=4,
                 no_map=False):
        super().__init__(height, bucket_size=bucket_size + s_num, block_size=block_size)
        self.union_size = 0
        self.no_map = no_map
        self.s_num = s_num
        self.a_num = a_num
        self.count = {}
        self.dummy_blocks = [dummy_block(self.block_size) for _ in range(self.height)]
        self.leaf_num = 2 ** (self.height - 1)
        self.position_map = {}  # position map, address to leaf index
        self.address_map = {}  # bucket position to address list
        self.bucket_size = bucket_size

        self.c_batch = c_batch
        self.maxStashSize = maxStashSize
        self.big_g = 0

        self.queryLog = []

        self.evictionLog = [-1 for _ in range(self.c_batch)]
        self.eviction_id = 0
        self.eviction_id_cache = [0 for _ in range(self.c_batch)]

        self.StashSet: List[list] = []  # [[(add, block), ...], ...]
        self.curr_stash = {}
        self.stash_valid = []  # True for accessible, False for accessed

        self.curr_DRL = {}  # address to block
        self.DR_LogSet: List[list] = []
        self.DRL_valid = []
        self.round = 0
        self.tempStashSet: List[list] = [[] for _ in range(self.c_batch)]

        self.init_logs()
        self.build_position_map()

        # For evaluation
        self.info = AccessInfo()

    def init_logs(self):
        for i in range(self.c_batch):
            tmp_stash = []
            self.stash_valid.append([True for _ in range(self.maxStashSize + self.c_batch)])
            for j in range(self.maxStashSize + self.c_batch):
                tmp_stash.append((urandom(16).hex(), dummy_block(self.block_size)))
            self.StashSet.append(tmp_stash)
        self.tempStashSet = deepcopy(self.StashSet)

        for i in range(self.c_batch):
            tmp_DRL = []
            self.DRL_valid.append([True for _ in range(2 * self.c_batch)])
            for j in range(2 * self.c_batch):
                tmp_DRL.append((urandom(16).hex(), dummy_block(self.block_size)))
            self.DR_LogSet.append(tmp_DRL)

    def print(self):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            for j in range(len(layer)):
                bucket = layer[j]
                position = 2 ** i + j - 1
                for kk in range(self.bucket_size):
                    block = bucket[kk]
                    address = self.address_map[position][kk]
                    print(f'{address}:{_data_str(block)}', end=', ')
                print('||', end=' ')
            print('\n')
        print()

    def build_position_map(self):
        if not self.no_map:
            for id in range((2 ** self.height - 1) * self.a_num * 3 // 4):
                _address = urandom(16).hex()
                self.position_map[_address] = -1

            for position in range(2 ** self.height - 1):
                self.address_map[position] = [-1] * (self.bucket_size + self.s_num)

                # Directly set the validation of dummies to 1
                for i in range(self.s_num):
                    self.address_map[position][i + self.bucket_size] = 1
        self.count = [0] * (2 ** self.height - 1)

    def read_log_set(self, address):
        current_id = ((self.round) // self.a_num) % self.c_batch
        for k in range(current_id):
            j = (current_id - 1 - k) % self.c_batch
            found_real = False
            for jj in range(2 * self.c_batch):
                tmp_add, block = self.DR_LogSet[j][jj]
                if address == tmp_add and address not in self.curr_DRL.keys():
                    if self.DRL_valid[j][jj]:
                        self.DRL_valid[j][jj] = False
                        self.curr_DRL[address] = block
                        found_real = True
                        break
            if not found_real:
                found_dummy = False
                for jj in range(self.c_batch, 2 * self.c_batch):
                    if self.DRL_valid[j][jj]:
                        self.DRL_valid[j][jj] = False
                        found_dummy = True
                        break
                if not found_dummy:
                    raise Exception("Run out of dummy in read_log_set")

        self.info.down_async += self.c_batch * self.block_size

    def write_log_set(self, _block):
        self.curr_DRL[self.op_address] = _block

        self.info.down_async += 2 * self.c_batch * self.block_size
        self.info.up_async += 2 * self.c_batch * self.block_size
        if self.query_id == self.c_batch - 1:
            j = self.big_g % self.c_batch
            log_j_1 = []
            for add, _block in self.curr_DRL.items():
                log_j_1.append((add, _block))
            if len(log_j_1) > self.c_batch:
                raise Exception("Overflow in write_log_set")
            while len(log_j_1) < 2 * self.c_batch:
                log_j_1.append((urandom(16).hex(), dummy_block(self.block_size)))

            for jj in range(2 * self.c_batch):
                self.DRL_valid[j][jj] = True

            self.DR_LogSet[j] = deepcopy(log_j_1)
            self.curr_DRL.clear()

    def read_stash_set(self):
        for k in range(self.c_batch):
            j = (self.eviction_id - 1 - k + self.c_batch) % self.c_batch
            tempStash = self.StashSet[j]
            found_real = False
            for jj in range(self.maxStashSize):
                address, block = tempStash[jj]
                if (self.op_address == address and address not in self.curr_stash.keys() and
                        self.stash_valid[j][jj]):
                    self.curr_stash[address] = block
                    self.stash_valid[j][jj] = False
                    found_real = True
                    break
            if not found_real:
                found_dummy = False
                for jj in range(self.maxStashSize, self.maxStashSize + self.c_batch):
                    if self.stash_valid[j][jj]:
                        self.stash_valid[j][jj] = False
                        found_dummy = True
                        break
                if not found_dummy:
                    raise Exception("Run out of dummy in read_stash_set")
        self.tempStashSet[self.eviction_id] = deepcopy(self.StashSet[self.eviction_id])

        self.info.down_async += (2 * self.c_batch + self.maxStashSize) * self.block_size
        self.info.up_async += (self.c_batch + self.maxStashSize) * self.block_size

    def readEST(self, k, leaf_id):
        result_set = []
        max_level = int(ceil(log2(k)) + 1)

        for i in range(self.height):
            tmp_position = super().get_position(leaf_id, i)
            if i < max_level:  # 说明当前的桶在EST中
                result_set.append(deepcopy(self[tmp_position]))

        return result_set

    def prepare_evict_process(self):
        # Prepare the data for evictions, commit eviction when there are c_batch requests
        # Different from original ConcurORAM, our batch processing allows directly writing eviction log
        self.evictionLog[self.eviction_id] = self.eviction_id

        evict_cnt = super().g_to_l(self.big_g)

        path = self.read_path(evict_cnt)

        path.reverse()  # 0 is root
        max_level = int(ceil(log2(self.c_batch)) + 1)
        tmp_id = (self.big_g - 1) % self.c_batch
        if tmp_id < 0:
            tempStash_i_1 = []
        else:
            tempStash_i_1 = deepcopy(self.tempStashSet[(self.big_g - 1) % self.c_batch])

        EST_buckets = self.readEST(self.c_batch, evict_cnt)  # 0 is root, k is closer to leaf

        for j in range(max_level):
            path[j] = deepcopy(EST_buckets[j])

        union = self.evict_to_path(path, tempStash_i_1, evict_cnt)

        tempStash_i = []
        for address, block in union.items():
            tempStash_i.append((address, block))

        if len(tempStash_i) > self.maxStashSize:
            raise Exception("Overflow in evict_to_path")

        while len(tempStash_i) < self.maxStashSize + self.c_batch:
            tempStash_i.append((urandom(16).hex(), dummy_block(self.block_size)))

        self.tempStashSet[self.eviction_id] = deepcopy(tempStash_i)

        # read blocks from woTree
        self.info.down_async += (self.height - max_level) * self.bucket_size * self.block_size
        # Process lock
        # read the tempStash and the blocks from EST
        self.info.down_sync += (self.c_batch + self.maxStashSize) * self.block_size
        self.info.down_sync += max_level * self.bucket_size + self.block_size
        # write back the tempStash and the blocks from EST
        self.info.up_sync += (self.c_batch + self.maxStashSize) * self.block_size
        self.info.up_sync += max_level * self.bucket_size + self.block_size
        # Process unlock
        self.info.up_async += (self.height - max_level) * self.bucket_size * self.block_size

        # read temp stash and EST, write temp stash and est
        self.info.rtt_sync += 2
        # read log and path, write path
        self.info.rtt_async += 2

    def evict_to_path(self, path, tempStash_i_1: List, leaf_id):
        union = {}
        # Union the data from DRL and tempStash_i_1
        # curr_DRL has the up-t-date data
        for add, block in self.DR_LogSet[self.eviction_id]:
            if add in self.position_map.keys():
                union[add] = block

        for (address, block) in tempStash_i_1:
            if (address in self.position_map.keys()) and (address not in union.keys()):
                union[address] = block

        for i in range(self.height):
            for j in range(self.bucket_size):
                position = super().get_position(leaf_id, i)
                tmp_address = self.address_map[position][j]
                if tmp_address != -1:
                    union[tmp_address] = path[i][j]
                    self.address_map[position][j] = -1

        for i in range(self.height - 1, -1, -1):
            position = super().get_position(leaf_id, i)

            tmp_stash = {}
            min_leaf, max_leaf = super().get_leave_range(position, self.height)
            for address, block in union.items():
                tmp_leaf = self.position_map[address]

                if min_leaf <= tmp_leaf <= max_leaf:
                    tmp_stash[address] = block

            tmp_len = min(len(tmp_stash), self.bucket_size)
            selected_address = random.sample(list(tmp_stash.keys()), tmp_len)

            for j in range(tmp_len):
                address_ = selected_address[j]
                block = tmp_stash[address_]
                self[position][j] = block
                self.address_map[position][j] = address_

                union.pop(address_)

            for j in range(tmp_len, self.bucket_size):
                self[position][j] = dummy_block(self.block_size)
                self.address_map[position][j] = -1

            for j in range(self.bucket_size, self.bucket_size + self.s_num):
                self[position][j] = self.dummy_blocks[i]
                self.address_map[position][j] = 1

            # Update record map after eviction
            if hasattr(self, 'record_map'):
                self.record_map[position] = (0, self.s_num)

        self.union_size = len(union)
        return union

    def evict_batch_commit(self):
        for idx in range(self.c_batch):
            eviction_id = self.eviction_id_cache[idx] % self.c_batch
            if eviction_id == -1:
                continue

            for jj in range(self.maxStashSize + self.c_batch):
                self.stash_valid[eviction_id][jj] = True
            for jj in range(2 * self.c_batch):
                self.DRL_valid[eviction_id][jj] = True
            self.evictionLog[eviction_id] = -1

            self.DR_LogSet[idx] = [(urandom(16).hex(), dummy_block(self.block_size)) for _ in range(2 * self.c_batch)]
            self.StashSet[eviction_id] = deepcopy(self.tempStashSet[eviction_id])

            if idx == self.c_batch - 1:
                self.curr_stash.clear()
                for add, block in self.StashSet[idx]:
                    if add in self.position_map.keys() and self.position_map[add] != -1:
                        self.curr_stash[add] = block
                self.info.down_sync += (self.c_batch + self.maxStashSize) * self.block_size

            # Update the data tree, all blocks are written back from woTree, thus no up_async
            self.info.rtt_sync += 1

    def read_concur_path(self, leaf_id):
        result_data = None
        for i in range(self.height):
            found_real = False
            tmp_position = super().get_position(leaf_id, i)
            for jj in range(self.bucket_size):
                tmp_address = self.address_map[tmp_position][jj]
                if tmp_address == self.op_address:
                    result_data = self[tmp_position][jj]
                    self.address_map[tmp_position][jj] = -1
                    found_real = True
                    break
            if not found_real:
                found_dummy = False
                for jj in range(self.bucket_size, self.bucket_size + self.s_num):
                    if self.address_map[tmp_position][jj] == 1:
                        self.address_map[tmp_position][jj] = 0
                        found_dummy = True
                        break
                if not found_dummy:
                    raise Exception("Run out of dummy in read_concur_path")
            self.count[tmp_position] += 1

            # Update record map, only when record map is added
            if hasattr(self, 'record_map'):
                bid = tmp_position
                if bid not in self.record_map.keys():
                    self.record_map[bid] = (1, self.s_num)
                else:
                    cnt, limit = self.record_map[bid]
                    self.record_map[bid] = (cnt + 1, min(self.s_num, limit))

        # download single block
        self.info.down_async += self.block_size
        return result_data

    def early_reshuffle(self, leaf_id):
        for i in range(self.height):
            position = super().get_position(leaf_id, i)
            if self.count[position] >= self.s_num:
                # For simplicity, reshuffle directly update the bucket, not shuffle with stash
                for jj in range(self.bucket_size, self.bucket_size + self.s_num):
                    self.address_map[position][jj] = 1
                self.count[position] = 0
                if hasattr(self, 'record_map') and position in self.record_map:
                    self.record_map[position] = (0, self.s_num)

                # download and upload single bucket, regarded as sync
                self.info.down_sync += self.bucket_size * self.block_size
                self.info.up_sync += self.bucket_size * self.block_size

            # Evict record, only when record map is added
            if hasattr(self, 'record_map'):
                bid = position
                cnt, limit = self.record_map[bid]
                if cnt >= limit:
                    for jj in range(self.bucket_size, self.bucket_size + self.s_num):
                        self.address_map[bid][jj] = 1
                    self.record_map[bid] = (0, self.s_num)
                    self.evict_record_access.add(self.eviction_id)
                    self.info.down_sync += (self.bucket_size + self.s_num) * self.block_size
                    self.info.up_sync += (self.bucket_size + self.s_num) * self.block_size

    def add_record_map(self, record_map):
        self.record_map = record_map

    def access(self, batch_requests):
        assert len(batch_requests) == self.c_batch
        self.batch_requests = batch_requests
        self.batch_ids = []
        self.info.clear()
        self.evict_record_access = set()

        for jj in range(self.c_batch):
            op, address, data = batch_requests[jj]
            self.query_id = len(self.queryLog) % self.c_batch
            self.batch_ids.append(self.query_id)
            if address in self.queryLog:
                self.queryLog.append(-1)
            else:
                self.queryLog.append(address)

        results = []
        for jj in range(self.c_batch):
            op, self.op_address, data = batch_requests[jj]
            self.query_id = self.batch_ids[jj]
            leaf_id = self.position_map[self.op_address]
            self.eviction_id = self.big_g % self.c_batch
            self.position_map[self.op_address] = random.randint(0, self.leaf_num - 1)

            if leaf_id == -1:
                leaf_id = random.randint(0, self.leaf_num - 1)

            # Evict the bucket accessed by Path ORAM, the stash here is empty, thus directly refresh the bucket
            if hasattr(self, 'record_map'):
                for i in range(self.height):
                    position = self.get_position(leaf_id, i)
                    if position not in self.record_map:
                        continue
                    cnt, limit = self.record_map[position]
                    if cnt >= limit:
                        for jj in range(self.bucket_size, self.bucket_size + self.s_num):
                            self.address_map[position][jj] = 1
                        self.record_map[position] = (0, self.s_num)
                        self.info.down_sync += (self.bucket_size + self.s_num) * self.block_size
                        self.info.up_sync += (self.bucket_size + self.s_num) * self.block_size

            result_data = self.read_concur_path(leaf_id)

            self.early_reshuffle(leaf_id)

            self.read_stash_set()
            if self.op_address in self.curr_stash.keys():
                result_data = self.curr_stash[self.op_address]

            self.read_log_set(self.op_address)
            if self.op_address in self.curr_DRL.keys():
                result_data = self.curr_DRL[self.op_address]

            if op == 'read' and result_data is None:
                self.print_meta()
                raise Exception(f"Data not found, address: {self.op_address}")

            if op == 'write':
                result_data = data

            self.curr_DRL[self.op_address] = result_data

            if len(self.curr_DRL) > self.c_batch:
                raise Exception(f"Overflow in curr_DRL, len:\t{len(self.curr_DRL)}")
            if len(self.curr_stash) > self.maxStashSize:
                raise Exception(f"Overflow in curr_stash, len:\t{len(self.curr_stash)}")

            results.append(result_data)

            self.write_log_set(result_data)

            # Write back from temp space
            if self.query_id == self.c_batch - 1:
                self.StashSet = deepcopy(self.tempStashSet)
                for j in range(self.c_batch):
                    for jj in range(self.maxStashSize + self.c_batch):
                        self.stash_valid[j][jj] = True
                    for jj in range(2 * self.c_batch):
                        self.DRL_valid[j][jj] = True
                self.queryLog = []

            self.info.rtt_sync += 1  # read query log, read log set and stash set
            self.info.rtt_async += 1  # write log set

            if self.round % self.a_num == (self.c_batch - 1):
                self.prepare_evict_process()
                self.eviction_id_cache[self.eviction_id] = self.big_g
                self.big_g += 1

            if self.round % (self.c_batch * self.a_num) == self.c_batch * self.a_num - 1:
                self.evict_batch_commit()

            self.round += 1

        if len(self.evict_record_access) > 0:
            self.info.evict_record_flag = True
        else:
            self.info.evict_record_flag = False

        return results, deepcopy(self.info)

    def print_meta(self):
        print(f'DR-logset:\t [')
        current_eviction_id = self.big_g % self.c_batch
        for i in range(self.c_batch):
            if i == current_eviction_id:
                print('>>', end='')
            print(f'\t{i}:\t[', end='')
            for add, block in self.DR_LogSet[i]:
                if add in self.position_map.keys():
                    print(f'{add}:{_data_str(block)}', end=', ')
            print(']')
        print(']')

        print(f'StashSet:\t [')
        for i in range(self.c_batch):
            print(f'\t{i}:\t[', end='')
            for add, block in self.StashSet[i]:
                if add in self.position_map.keys():
                    print(f'{add}:{_data_str(block)}', end=', ')
            print(']')
        print(']')

        print('stash:\t[', end='')
        for add, block in self.curr_stash.items():
            print(f'{add}:{_data_str(block)}', end=', ')
        print(']')

        print('DRL:\t[', end='')
        for add, block in self.curr_DRL.items():
            print(f'{add}:{_data_str(block)}', end=', ')
        print(']')

        self.print()
        print(f'=====================================')


if __name__ == "__main__":
    test_factor = 10
    oram_height = 7
    c_batch = 8  # c_batch = a_num
    bucket_size = 8
    a_num = 8
    slum_num = 12
    concur = ConcurORAM(height=oram_height, bucket_size=bucket_size,
                        c_batch=c_batch, a_num=a_num, s_num=slum_num)
    p_map = concur.position_map
    maxsize = -1

    stash_count = {}
    bucket_load = []
    real_datasets = {}
    for i in range(2 ** test_factor):
        batch_requests = []
        while len(batch_requests) < c_batch:
            if random.random() < 0.5:
                address = random.choice(list(p_map.keys()))
                data = dummy_block(concur.block_size)
                real_datasets[address] = data
                batch_requests.append(('write', address, data))
            else:
                if len(real_datasets) == 0:
                    continue
                address = random.choice(list(real_datasets.keys()))
                real = real_datasets[address]
                dummy = dummy_block(concur.block_size)
                batch_requests.append(('read', address, real))

        results, _ = concur.access(batch_requests)
        for jj in range(c_batch):
            op, add, data = batch_requests[jj]
            if op == 'read':
                try:
                    assert data == results[jj]
                except AssertionError:
                    raise Exception(f"Error:\tresult:\t {add}:{results[jj]}\n"
                                    f"real:\t{data}")

        if concur.union_size in stash_count.keys():
            stash_count[concur.union_size] += 1
        else:
            stash_count[concur.union_size] = 1

    for i in range(oram_height):
        cnt = 0
        for bios in range(2 ** i):
            id = 2 ** i + bios - 1
            cnt += concur.address_map[id].count(-1)
        bucket_load.append(1 - cnt / (2 ** i * concur.bucket_size))

    stash_count_sorted = sorted(stash_count.items())
    for key, value in stash_count_sorted:
        print(f"{key}: {value}, ", end='')
    print()

    cnt = 0
    cnt_ = 0
    for add, block in concur.curr_stash.items():
        if add in real_datasets.keys():
            cnt += 1
        if add in concur.position_map.keys():
            cnt_ += 1

    print(bucket_load)
