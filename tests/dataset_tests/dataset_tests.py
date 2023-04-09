# A workaround for import error
import sys, os
from pathlib import Path
import unittest
import logging
p = Path(__file__).parents[2]
p = os.path.join(p, 'asociita')
sys.path.insert(0, p)

from asociita.datasets.pytorch.fetch_data import load_dataset


class Dataset_Tests(unittest.TestCase):

    def load_mnist_test(self):
        dataset_name = 'mnist'
        settings = {}
        train_data, test_data = load_dataset(
            dataset_name=dataset_name,
            settings=settings)
        self.assertIsNotNone(train_data)
        self.assertIsNotNone(test_data)
        print("SHOWING DOWLONADED DATASET")
        print(test_data)
        print(train_data)
        print("Initialization tests passed successfully")


if __name__ == "__main__":
    test_instance = Dataset_Tests()
    test_instance.load_mnist_test()