from asociita.components.settings.settings import Settings
from asociita.datasets.fetch_data import load_data
import unittest

class HetDataGenerationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
        "dataset_name" : "mnist",
        "split_type" : "heterogeneous_size",
        "shards": 10,
        "local_test_size": 0.2,
        "transformations": {},
        "imbalanced_clients": {},
        "save_dataset": False,
        "save_transformations": False,
        "save_blueprint": False,
        "agents": 10}
    
    def testGeneration(self) -> None:
        data = load_data(self.config)
        self.assertTrue(data)

def unit_test_hetgen():
    case = HetDataGenerationTestCase()
    case.setUp()
    case.testGeneration()
    print("All unit tests for Heterogeneous data generation has been passed.")
    return 0