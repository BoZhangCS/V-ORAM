import sys, os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import unittest
from random import random, choice
from src.V_ORAM import V_ORAM
from src.BTree import BLOCK_SIZE
from os import urandom


class TestOST_Ring_Path(unittest.TestCase):
    def setUp(self):
        self.c_batch = 8
        self.v_oram = V_ORAM(10, c_batch=self.c_batch, a_num=8, bucket_size=8, s_num=12)
        self.p_map = self.v_oram.curr_ORAM.position_map
        self.repeat = 2 ** 8
        self.total_periods = 9
        self.real_datasets = {}

    def tearDown(self):
        del self.v_oram
        del self.p_map
        del self.repeat
        del self.real_datasets
        del self.total_periods

    def test_read_write(self):
        real_datasets = {}
        for _ in range(self.total_periods):
            if _ % 2 == 1:
                sid = 'path'
            else:
                sid = 'ring'
            for i in range(self.repeat):
                if random() < 0.5:
                    address = choice(list(self.v_oram.curr_ORAM.position_map.keys()))
                    data = urandom(BLOCK_SIZE)
                    self.v_oram.access('write', address, data, sid)
                    real_datasets[address] = data
                else:
                    if len(real_datasets) == 0:
                        i -= 1
                        continue
                    address = choice(list(real_datasets.keys()))
                    data = real_datasets[address]
                    result, _ = self.v_oram.access('read', address, urandom(BLOCK_SIZE), sid)
                    try:
                        assert (result == data)
                    except AssertionError:
                        print(f'Error: {result[:4].hex()} != {data[:4].hex()}')
                        raise AssertionError


if __name__ == '__main__':
    unittest.main()
