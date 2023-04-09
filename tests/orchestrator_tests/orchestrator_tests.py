# A workaround for import error
import sys, os
from pathlib import Path
import unittest
import logging
p = Path(__file__).parents[2]
p = os.path.join(p, 'asociita')
sys.path.insert(0, p)


from asociita.components.orchestrator.orchestrator import Orchestrator
from asociita.models.pytorch.mnist import MnistNet
from asociita.datasets.pytorch.fetch_data import load_mnist
class Orchestrator_tests(unittest.TestCase):

    def init_test(self):
        settings = dict()
        test_orchestrator = Orchestrator(settings=settings)
        self.assertEqual(0, test_orchestrator.state)
    
    
    def load_model(self):
        # Creates Orchestrator instance
        settings = dict()
        test_orchestrator = Orchestrator(settings=settings)
        self.assertEqual(0, test_orchestrator.state)
        
        # Loads model onto the orchestrator
        model = MnistNet()
        test_orchestrator.load_model(model)
        self.assertIsNotNone(test_orchestrator.model)
        self.assertEqual(0, test_orchestrator.state)
    

    def load_data(self):
        # Creates Orchestrator instance
        settings = dict()
        test_orchestrator = Orchestrator(settings=settings)
        self.assertEqual(0, test_orchestrator.state)
        
        # Loads model onto the orchestrator
        model = MnistNet()
        test_orchestrator.load_model(model)
        self.assertIsNotNone(test_orchestrator.model)
        self.assertEqual(0, test_orchestrator.state)

        # Loads the dataset
        val_data = load_mnist()
        test_orchestrator.load_data(val_data)
        self.assertIsNotNone(test_orchestrator.validation_data)
        self.assertEqual(0, test_orchestrator.state)



if __name__ == "__main__":
    tests_inst = Orchestrator_tests()
    tests_inst.init_test()
    tests_inst.load_model()
    tests_inst.load_data()