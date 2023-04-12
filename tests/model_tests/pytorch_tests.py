# A workaround for import error
import sys, os
from pathlib import Path
import unittest
import logging
p = Path(__file__).parents[2]
p = os.path.join(p, 'asociita')
sys.path.insert(0, p)

from asociita.models.pytorch.federated_model import FederatedModel
from asociita.datasets.fetch_data import load_data
from asociita.models.pytorch.mnist import MnistNet


class PyTorch_Tests(unittest.TestCase):

    def test_init(self):
        settings_data = {"dataset_name": 'mnist',
                         "split_type": "random_uniform",
                         "shards": 10,
                         "local_test_size": 0.2,}
        settings_node = {
            "optimizer": "RMS",
            "batch_size": 32,
            "learning_rate": 0.1}
        data = load_data(settings=settings_data)
        model = MnistNet()

        federated_model = FederatedModel(settings=settings_node,
                                         local_dataset=data,
                                         net=model)
        print(federated_model.trainloader)
        print(federated_model.testloader)
        
        print("Initialization tests passed successfully")


if __name__ == "__main__":
    test_instance = PyTorch_Tests()
    test_instance.test_init()