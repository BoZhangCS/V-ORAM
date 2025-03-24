from copy import deepcopy
from random import randint, sample, random, choice

from src.AccessInfo import AccessInfo
from src.BTree import BTree, dummy_block
from os import urandom

class Path_ORAM(BTree):
    """
    A Path ORAM (Oblivious RAM) implementation that extends the BTree class.

    Attributes:
        leaf_num (int): Number of leaves in the ORAM tree.
        stash (dict): A dictionary that stores the client cached blocks.
        position_map (dict): A dictionary that maps the block address to its leaf index.
        address_map (dict): A dictionary that maps the block location on the tree to its address.
        read_buckets (list): A list used to record the buckets read during access.
        evicted_buckets (list): A list used to record the buckets evicted during access.
        info (AccessInfo): An object used to record access information for evaluation.

    Methods:
        __init__(height, bucket_size=8, block_size=4096, no_map=False):
            Initializes the Path_ORAM instance with the given parameters.

        build_position_map():
            Initialize the position map and address map for the ORAM tree.
            For position map, -1 denotes un-accessed blocks, otherwise, the leaf index of the block.
            For address map, -1 denotes un-accessed blocks, otherwise, the address of the block.

        access(op, address, data):
            Performs an access operation (read or write) on the ORAM tree.

        eviction(x):
            Evicts blocks from the stash to the ORAM tree.
    """
    def __init__(self, height, bucket_size=8, block_size=4096, no_map=False):
        super().__init__(height, bucket_size=bucket_size, block_size=block_size)
        self.leaf_num = 2 ** (self.height - 1)
        self.stash = {}
        self.position_map = {}  # position map in paper, address to leaf index
        self.address_map = {}  # from block location to address list

        # This part is used for record map, the buckets being read and evicted in this access
        self.read_buckets = []
        self.evicted_buckets = []

        # For evaluation
        self.info = AccessInfo()

        if not no_map:
            self.build_position_map()

    def build_position_map(self):
        for id in range(2 ** (self.height - 1) * self.bucket_size):
            _address = urandom(16).hex()
            self.position_map[_address] = -1

        for position in range(2 ** self.height - 1):
            self.address_map[position] = [-1] * self.bucket_size

    def access(self, op, address, data):
        self.read_buckets = []
        self.evicted_buckets = []
        self.info.clear()

        x = self.position_map[address]
        if x == -1:
            # First time access
            x = randint(0, self.leaf_num - 1)

        self.position_map[address] = randint(0, self.leaf_num - 1)

        path = self.read_path(x)

        tmp_position = 2 ** (self.height - 1) + x - 1
        for bucket in path:
            for bios in range(self.bucket_size):
                tmp_address = self.address_map[tmp_position][bios]
                if tmp_address != -1:
                    # Retrieve the valid blocks
                    self.stash[tmp_address] = bucket[bios]
                    self.address_map[tmp_position][bios] = -1

                    # For record map
                    self.read_buckets.append(tmp_position)

            tmp_position = (tmp_position - 1) // 2

        result_data = None
        if op == 'write':
            self.stash[address] = data

        if address in self.stash.keys():
            result_data = self.stash[address]

        self.eviction(x)

        self.info.down_sync += self.height * self.bucket_size * self.block_size
        self.info.up_sync += self.height * self.bucket_size * self.block_size
        self.info.rtt_sync += 2

        return result_data, deepcopy(self.read_buckets), deepcopy(self.evicted_buckets), deepcopy(self.info)

    def eviction(self, x):
        # Eviction
        tmp_position = 2 ** (self.height - 1) + x - 1
        for l in range(self.height - 1, -1, -1):
            # Randomly select the block from stash
            tmp_stash = {}
            min_leaf, max_leaf = super().get_leave_range(tmp_position, self.height)
            for address, block in self.stash.items():
                required_leaf = self.position_map[address]
                if required_leaf >= min_leaf and required_leaf <= max_leaf:
                    tmp_stash[address] = block

            tmp = min(len(tmp_stash), self.bucket_size)
            selected_address = sample(list(tmp_stash.keys()), tmp)

            for j in range(tmp):
                self.stash.pop(selected_address[j])
                block = tmp_stash[selected_address[j]]
                self[tmp_position][j] = deepcopy(block)

                self.address_map[tmp_position][j] = selected_address[j]

            # Pad dummies
            for j in range(tmp, self.bucket_size):
                self.address_map[tmp_position][j] = -1
                self[tmp_position][j] = dummy_block(self.block_size)

            # For record map
            self.evicted_buckets.append(tmp_position)
            tmp_position = (tmp_position - 1) // 2


if __name__ == '__main__':
    test_factor = 12
    oram_height = 5
    path_oram = Path_ORAM(oram_height)

    stash_count = {}
    bucket_load = []
    real_datasets = {}
    for i in range(2 ** test_factor):
        if random() < 0.5:
            address = choice(list(path_oram.position_map.keys()))
            data = urandom(4096)
            path_oram.access('write', address, data)
            real_datasets[address] = data
        else:
            if len(real_datasets) == 0:
                i -= 1
                continue
            address = choice(list(real_datasets.keys()))
            data = real_datasets[address]
            result = path_oram.access('read', address, urandom(4096))
            assert (result == data)

        if len(path_oram.stash) in stash_count.keys():
            stash_count[len(path_oram.stash)] += 1
        else:
            stash_count[len(path_oram.stash)] = 1

    for i in range(oram_height):
        cnt = 0
        for bios in range(2 ** i):
            id = 2 ** i + bios - 1
            cnt += path_oram.address_map[id].count(-1)
        bucket_load.append(1 - cnt / (2 ** i * path_oram.bucket_size))

    print(len(path_oram.stash))
    for add, data in path_oram.stash.items():
        print(add, data[:4].hex())
    print(stash_count)

    print(bucket_load)
