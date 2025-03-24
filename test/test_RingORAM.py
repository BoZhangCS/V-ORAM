import sys, os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import unittest
import random
from src.Ring_ORAM import Ring_ORAM
from os import urandom


class TestRingORAM(unittest.TestCase):
    def setUp(self):
        self.ring_oram = Ring_ORAM(8)
        self.p_map = self.ring_oram.position_map
        self.repeat = 2 ** 10
        self.real_datasets = {}

    def tearDown(self):
        del self.ring_oram
        del self.p_map
        del self.repeat
        del self.real_datasets

    def test_read_write(self):
        # print(self.address)
        # print(self.path_oram.stash.keys())
        for i in range(self.repeat):
            if random.random() < 0.5:
                address = random.randint(0, len(self.ring_oram.position_map) - 1)
                data = urandom(4096)
                self.ring_oram.access('write', address, data)
                self.real_datasets[address] = data
            else:
                if len(self.real_datasets) == 0:
                    i -= 1
                    continue
                address = random.choice(list(self.real_datasets.keys()))
                data = self.real_datasets[address]
                result, _, _, _ = self.ring_oram.access('read', address, urandom(4096))
                assert (result == data)


if __name__ == '__main__':
    unittest.main()
