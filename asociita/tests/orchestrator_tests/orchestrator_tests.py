from asociita.components.orchestrator.orchestrator import Orchestrator
from asociita.models.pytorch.mnist import MnistNet
import unittest

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


if __name__ == "__main__":
    tests_inst = Orchestrator_tests()
    tests_inst.init_test()
    tests_inst.load_model()