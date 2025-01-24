import sys, os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import unittest
import random

from src.ConcurORAM import ConcurORAM
from os import urandom


class TestConcurORAM(unittest.TestCase):
    def setUp(self):
        self.c_batch = 8
        self.concur = ConcurORAM(10, c_batch=self.c_batch, a_num=8, bucket_size=8, s_num=12)
        self.p_map = self.concur.position_map
        self.repeat = 2 ** 8
        self.real_datasets = {}

    def tearDown(self):
        del self.concur
        del self.p_map
        del self.repeat
        del self.real_datasets

    def test_read_write(self):
        # print(self.address)
        # print(self.path_oram.stash.keys())
        for i in range(self.repeat):
            batch_requests = []
            while len(batch_requests) < self.c_batch:
                if random.random() < 0.5:
                    address = random.choice(list(self.p_map.keys()))
                    data = urandom(4096)
                    self.real_datasets[address] = data
                    batch_requests.append(('write', address, data))
                else:
                    if len(self.real_datasets) == 0:
                        continue
                    address = random.choice(list(self.real_datasets.keys()))
                    real = self.real_datasets[address]
                    batch_requests.append(('read', address, real))

            results = self.concur.access(batch_requests)
            for jj in range(self.c_batch):
                op, add, data = batch_requests[jj]
                if op == 'read':
                    try:
                        assert data == results[0][jj]
                    except AssertionError:
                        raise Exception(f"Error:\tresult:\t {add}:{results[0][jj][:4].hex()}\n"
                                        f"real:\t{data[:4].hex()}")


if __name__ == '__main__':
    unittest.main()
