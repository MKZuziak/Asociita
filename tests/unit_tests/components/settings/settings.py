from asociita.components.settings.settings import Settings
import unittest

class SettingsInitTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "orchestrator": {
                "iterations": 10,
                "number_of_nodes": 5,
                "local_warm_start": False,
                "sample_size": 2,
                "metrics_save_path": "None",
                "nodes": [
                0,
                1,
                2,
                3,
                4]
            },
            "nodes":{
            "local_epochs": 3,
            "model_settings": {
                "optimizer": "RMS",
                "batch_size": 64,
                "learning_rate": 0.0031622776601683794}
                }
        }
    
    def testInit(self) -> None:
        self.settings = Settings(initialization_method='dict',
                                 dict_settings = self.config)
    
    def testPropertyRetrive(self) -> None:
        self.assertEqual(self.settings.iterations, 10)
        self.assertEqual(self.settings.number_of_nodes, 5)
        self.assertEqual(self.settings.local_warm_start, False)
        self.assertEqual(self.settings.metrics_save_path, 'None')
        self.assertEqual(self.settings.optimizer, 'RMS')
        self.assertEqual(self.settings.batch_size, 64)
        self.assertEqual(self.settings.sample_size, 2)
        self.assertEqual(self.settings.lr, 0.0031622776601683794)
    

def unit_test_settings():
    case = SettingsInitTestCase()
    case.setUp()
    case.testInit()
    case.testPropertyRetrive()
    print("All unit tests for Setting Object Initialization has been passed.")

unit_test_settings()