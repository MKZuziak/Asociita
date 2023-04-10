# A workaround for import error
import sys, os
from pathlib import Path
import unittest
import logging
p = Path(__file__).parents[2]
p = os.path.join(p, 'asociita')
sys.path.insert(0, p)

from asociita.datasets.fetch_data import load_data


class Dataset_Tests(unittest.TestCase):

    def load_mnist_test(self):
        settings = {
            "dataset_name" : 'mnist',
            "split_type" : 'random_uniform',
            "shards": 10,
            "local_test_size": 0.2}

        data = load_data(settings=settings)
        
        self.assertIsNotNone(data)
        print("SHOWING DOWLONADED DATASET")
        print(data)
        print("Initialization tests passed successfully")


if __name__ == "__main__":
    test_instance = Dataset_Tests()
    test_instance.load_mnist_test()