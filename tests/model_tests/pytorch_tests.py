# A workaround for import error
import sys, os
from pathlib import Path
from collections import OrderedDict
import unittest
import logging
import copy
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
                         "local_test_size": 0.5,}
        settings_node = {
            "optimizer": "RMS",
            "batch_size": 32,
            "learning_rate": 0.001}
        data = load_data(settings=settings_data)
        data = data[1][1]
        model = MnistNet()

        self.federated_model = FederatedModel(settings=settings_node,
                                         local_dataset=data,
                                         net=model,
                                         node_name=0)
        print(self.federated_model.trainloader)
        print(self.federated_model.testloader)

        self.assertIsNotNone(self.federated_model)
        self.assertIsNotNone(self.federated_model.trainloader)
        self.assertIsNotNone(self.federated_model.testloader)
        self.assertEqual(0, self.federated_model.node_name)
        print("Initialization tests passed successfully")
    

    def test_get_weights(self):
        weights = self.federated_model.get_weights_list()
        self.assertIs(type(weights), list)
        weights = self.federated_model.get_weights()
        self.assertIs(type(weights), OrderedDict)
        print("Weights has been received successfully")
    

    def test_update_weights(self):
        weights = self.federated_model.get_weights()
        new_weights = copy.deepcopy(weights)
        self.federated_model.update_weights(new_weights)
        weights = self.federated_model.get_weights()
        self.assertIs(type(weights), OrderedDict)
        print("Weights has been updated successfully")
    

    def test_train(self):
        for _ in range(20):
            loss, acc = self.federated_model.train()
            metrics = self.federated_model.evaluate_model()
            print(metrics)


if __name__ == "__main__":
    test_instance = PyTorch_Tests()
    test_instance.test_init()
    test_instance.test_get_weights()
    test_instance.test_update_weights()
    test_instance.test_train()