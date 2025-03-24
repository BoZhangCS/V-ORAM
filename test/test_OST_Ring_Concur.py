import sys, os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import unittest
from random import random, choice, randint
from src.V_ORAM import V_ORAM
from src.BTree import dummy_block
from os import urandom


class Test_Ring_Concur(unittest.TestCase):
    def setUp(self):
        self.c_batch = 8
        self.v_oram = V_ORAM(10, c_batch=self.c_batch, a_num=8, bucket_size=8, s_num=12)
        self.p_map = self.v_oram.curr_ORAM.position_map
        self.repeat = 2 ** 10
        self.total_periods = 8
        self.real_datasets = {}

    def tearDown(self):
        del self.v_oram
        del self.p_map
        del self.repeat
        del self.real_datasets
        del self.total_periods

    def test_read_write(self):
        p_map = self.v_oram.curr_ORAM.position_map
        real_datasets = {}
        for _ in range(self.total_periods):
            if _ % 2 == 0:
                sid = 'concur'
            else:
                sid = 'ring'

            if sid != 'concur':
                for i in range(self.repeat):
                    if random() < 0.5:
                        address = randint(0, len(p_map) - 1)
                        data = urandom(4096)
                        self.v_oram.access('write', address, data, sid)
                        real_datasets[address] = data
                    else:
                        if len(real_datasets) == 0:
                            i -= 1
                            continue
                        address = choice(list(real_datasets.keys()))
                        data = real_datasets[address]
                        result, _ = self.v_oram.access('read', address, urandom(4096), sid)
                        try:
                            assert (result == data)
                        except Exception as e:
                            print(e)
            else:
                batch_requests = []
                test_num = self.repeat
                for i in range(test_num):
                    if random() < 0.5:
                        address = randint(0, len(p_map) - 1)
                        data = dummy_block()
                        real_datasets[address] = data
                        batch_requests.append(('write', address, data))
                        results, _ = self.v_oram.access('write', address, data, 'concur')
                    else:
                        if len(real_datasets) == 0:
                            continue
                        address = choice(list(real_datasets.keys()))
                        real = real_datasets[address]
                        batch_requests.append(('read', address, real))
                        results, _ = self.v_oram.access('read', address, real, 'concur')

                    if i == test_num - 1 and len(batch_requests) < self.c_batch:
                        while len(batch_requests) < self.c_batch:
                            address = choice(list(real_datasets.keys()))
                            real = real_datasets[address]
                            batch_requests.append(('read', address, real))
                            results, _ = self.v_oram.access('read', address, real, 'concur')
                        batch_requests = []

                    if len(batch_requests) == self.c_batch:
                        for jj in range(self.c_batch):
                            op, add, data = batch_requests[jj]
                            if op == 'read':
                                try:
                                    assert data == results[jj]
                                except Exception:
                                    print(type(data), type(results[jj]))
                                    raise Exception(f"Error:\tresult:\t {add}:{results[jj]}\n"
                                                    f"real:\t{data}")
                        batch_requests = []


if __name__ == '__main__':
    unittest.main()
