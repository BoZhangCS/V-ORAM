from copy import deepcopy
from os import urandom
from random import randint, choice, sample, random

from src.AccessInfo import AccessInfo
from src.BTree import BTree, dummy_block


class Ring_ORAM(BTree):
    """
    A Ring ORAM (Oblivious RAM) implementation that extends the BTree class.

    Attributes:
        dummy_blocks (list): List of dummy blocks for each level of the tree.
        s_num (int): Number of dummy blocks in each bucket.
        a_num (int): Number of constant A in Ring ORAM, evict_path is invoked for every A accesses.
        leaf_num (int): Number of leaves in the ORAM tree.
        bucket_size (int): Size of each bucket.
        stash (dict): Dictionary to store the client cached blocks.
        position_map (dict): Dictionary mapping block addresses to its leaf indices.
        address_map (dict): Dictionary mapping bucket positions to its address.
        round (int): Current round of operations.
        big_g (int): Global counter for evictions.
        count (dict): Dictionary to count accesses to each bucket.
        read_buckets (list): List to record the buckets read during accesses.
        evicted_buckets (list): List to record the buckets evicted during accesses.
        info (AccessInfo): Object to record access information for evaluation.

    Methods:
        __init__(height, bucket_size=8, block_size=4096, s_num=12, a_num=8):
            Initializes the Ring_ORAM instance with the given parameters.

        build_position_map():
            Builds the position map and address map for the ORAM tree.

        evict_path():
            Evicts blocks from the stash to the ORAM tree.

        _write_bucket(position, i):
            Writes blocks from the stash to a specific bucket.

        early_reshuffle(l):
            Performs an early reshuffle for a given leaf ID.

        read_ring_path(l, address):
            Reads the ring path for a given leaf ID and address.

        access(op, address, data_prime):
            Performs an access operation (read or write) on the ORAM tree.
    """
    def __init__(self, height, bucket_size=8, block_size=4096, s_num=12, a_num=8):
        super().__init__(height, bucket_size=bucket_size + s_num, block_size=block_size)
        self.dummy_blocks = [dummy_block(self.block_size) for _ in range(self.height)]
        self.s_num = s_num
        self.a_num = a_num
        self.leaf_num = 2 ** (self.height - 1)
        self.bucket_size = bucket_size

        self.stash = {}
        self.position_map = {}
        self.address_map = {}  # -1 denotes un-accessed blocks, 0 denotes accessed blocks, 1 denotes un-accessed dummy
        self.round = 0
        self.big_g = 0  # constant G in the paper
        self.count = {}  # record the number of accesses

        # Record map metadata
        self.read_buckets = []
        self.evicted_buckets = []

        # For evaluate
        self.info = AccessInfo()

        self.build_position_map()

    def build_position_map(self):
        for id in range(2 ** (self.height - 1) * self.a_num):
            address = urandom(16).hex()
            self.position_map[address] = -1

        for position in range(2 ** self.height - 1):
            self.address_map[position] = [-1] * (self.bucket_size + self.s_num)
            self.count[position] = 0

            # Directly set the validation of dummies to 1
            for i in range(self.s_num):
                self.address_map[position][i + self.bucket_size] = 1

    def evict_path(self):
        l = super().g_to_l(self.big_g)
        self.big_g = self.big_g + 1

        for i in range(self.height):
            position = super().get_position(l, i)
            for j in range(self.bucket_size):
                address = self.address_map[position][j]
                if address != -1:
                    self.stash[address] = self[position][j]
                    self.address_map[position][j] = -1

        for i in range(self.height - 1, -1, -1):
            position = super().get_position(l, i)
            self._write_bucket(position, i)
            self.count[position] = 0

            self.evicted_buckets.append(position)

        # For eval, upload is already calculated in write_bucket
        self.info.down_sync += self.height * (self.bucket_size + self.s_num) * self.block_size

    def _write_bucket(self, position, i):
        tmp_stash = {}
        min_leaf, max_leaf = super().get_leave_range(position, self.height)

        for address, block in self.stash.items():
            leaf_id = self.position_map[address]
            if leaf_id >= min_leaf and leaf_id <= max_leaf:
                tmp_stash[address] = deepcopy(block)

        tmp_len = min(len(tmp_stash), self.bucket_size)
        selected_address = sample(list(tmp_stash.keys()), tmp_len)

        for j in range(tmp_len):
            self.stash.pop(selected_address[j])
            block = tmp_stash[selected_address[j]]
            self[position][j] = deepcopy(block)

            self.address_map[position][j] = selected_address[j]

        for j in range(tmp_len, self.bucket_size):
            self.address_map[position][j] = -1
            self[position][j] = dummy_block(self.block_size)

        for j in range(self.bucket_size, self.bucket_size + self.s_num):
            self.address_map[position][j] = 1
            self[position][j] = deepcopy(self.dummy_blocks[i])

        # For eval, merely upload here
        self.info.up_sync += (self.bucket_size + self.s_num) * self.block_size

    def early_reshuffle(self, l):
        evicted = False
        for i in range(self.height):
            position = super().get_position(l, i)
            if self.count[position] >= self.s_num:
                evicted = True
                for j in range(self.bucket_size):
                    address = self.address_map[position][j]
                    if address != -1:
                        self.stash[address] = self[position][j]
                        self.address_map[position][j] = -1
                self._write_bucket(position, i)
                self.count[position] = 0

                # For record map
                self.evicted_buckets.append(position)
                self.info.down_sync += (self.bucket_size + self.s_num) * self.block_size
                self.info.up_sync += (self.bucket_size + self.s_num) * self.block_size
        return evicted

    def read_ring_path(self, l, address):
        blocks = []
        result_data = None
        for i in range(self.height):
            position = super().get_position(l, i)
            address_list = self.address_map[position]
            found = False
            for j in range(self.bucket_size):
                if address == address_list[j]:
                    result_data = deepcopy(self[position][j])
                    self.address_map[position][j] = -1
                    found = True
                    break
            if not found:
                # Randomly return a valid dummy, overflow is there is no valid dummy
                flag = None
                for j in range(self.bucket_size, self.bucket_size + self.s_num):
                    if self.address_map[position][j] == 1:
                        blocks.append(self[position][j])
                        self.address_map[position][j] = 0
                        flag = True
                        break
                if flag is None:
                    raise Exception('Overflow happened when finding valid dummy blocks')
            self.count[position] += 1

            # For record map
            self.read_buckets.append(position)

        # For eval
        self.info.down_sync += self.block_size
        # Simplify XOR, directly return the result
        return result_data

    def access(self, op, address, data_prime):
        self.read_buckets = []
        self.evicted_buckets = []
        self.info.clear()

        l_prime = randint(0, self.leaf_num - 1)
        l = self.position_map[address]
        self.position_map[address] = l_prime
        first_write = False

        if l == -1:
            l = randint(0, self.leaf_num - 1)
            first_write = True

        data = self.read_ring_path(l, address)

        if data is None:
            if not first_write:
                try:
                    data = self.stash[address]
                except Exception:
                    print('Not found the data in stash')
                    data = dummy_block(self.block_size)
            elif op == 'read':
                raise Exception('Read data before write')

        if op == 'write':
            data = deepcopy(data_prime)

        self.stash[address] = data

        if address in self.stash.keys():
            result_data = deepcopy(self.stash[address])
        else:
            raise Exception('Data not found in stash')

        self.round = (self.round + 1) % self.a_num

        if self.round == 0:
            self.evict_path()

        evicted = self.early_reshuffle(l)

        self.info.rtt_sync += 1  # For read ring path
        if evicted or self.round == 0:
            self.info.rtt_sync += 1  # For eviction, only need one rtt

        return result_data, deepcopy(self.read_buckets), deepcopy(self.evicted_buckets), deepcopy(self.info)


if __name__ == '__main__':
    test_factor = 12
    oram_height = 5
    ring_oram = Ring_ORAM(oram_height)
    p_map = ring_oram.position_map
    maxsize = -1

    stash_count = {}
    bucket_load = []
    real_datasets = {}
    for i in range(2 ** test_factor):
        if random() < 0.5:

            address = choice(list(p_map.keys()))
            data = dummy_block(4096)
            ring_oram.access('write', address, data)
            real_datasets[address] = data

        else:
            if len(real_datasets) == 0:
                i -= 1
                continue

            address = choice(list(real_datasets.keys()))
            data = real_datasets[address]
            result = ring_oram.access('read', address, urandom(4096))

            assert (result == data)

        if len(ring_oram.stash) in stash_count.keys():
            stash_count[len(ring_oram.stash)] += 1
        else:
            stash_count[len(ring_oram.stash)] = 1

    for i in range(oram_height):
        cnt = 0
        for bios in range(2 ** i):
            id = 2 ** i + bios - 1
            cnt += ring_oram.address_map[id].count(-1)
        bucket_load.append(1 - cnt / (2 ** i * ring_oram.bucket_size))

    stash_count_sorted = sorted(stash_count.items())
    for key, value in stash_count_sorted:
        print(f"{key}: {value}, ", end='')
    print()

    print(bucket_load)
