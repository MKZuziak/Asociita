from asociita.components.settings.settings import Settings
from asociita.components.settings.init_settings import init_settings
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
                'enable_archiver': False,
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
        self.settings = init_settings(orchestrator_type='general',
                                      allow_default=True,
                                      initialization_method='dict',
                                      dict_settings=self.config)

    def testPropertyRetrive(self) -> None:
        self.assertEqual(self.settings.iterations, 10)
        self.assertEqual(self.settings.number_of_nodes, 5)
        self.assertEqual(self.settings.sample_size, 2)
        self.assertEqual(self.settings.local_warm_start, False)
        self.assertEqual(self.settings.optimizer, 'RMS')
        self.assertEqual(self.settings.batch_size, 64)
        self.assertEqual(self.settings.lr, 0.0031622776601683794)


class EmptyInitTestCase(SettingsInitTestCase):
        def setUp(self) -> None:
             self.config = dict()
        
        def testInit(self) -> None:
             return super().testInit()
        
        def testPropertyRetrive(self) -> None:
            self.assertEqual(self.settings.iterations, 10)
            self.assertEqual(self.settings.number_of_nodes, 10)
            self.assertEqual(self.settings.sample_size, 5)
            self.assertEqual(self.settings.local_warm_start, False)
            self.assertEqual(self.settings.optimizer, 'RMS')
            self.assertEqual(self.settings.batch_size, 32)
            self.assertEqual(self.settings.lr, 0.001)


class MixedInitTestCase(EmptyInitTestCase):
     def setUp(self) -> None:
        self.config = {
            "orchestrator": {
                "iterations": 10,
                "sample_size": 5,
                "metrics_save_path": "None",
                'enable_archiver': False,
            },
            "nodes":{
            "local_epochs": 3}
        }
    

def unit_test_settings():
    case = SettingsInitTestCase()
    case.setUp()
    case.testInit()
    case.testPropertyRetrive()
    
    case2 = EmptyInitTestCase()
    case2.setUp()
    case2.testInit()
    case2.testPropertyRetrive()
    
    case3 = MixedInitTestCase()
    case3.setUp()
    case3.testInit()
    case3.testPropertyRetrive()
    
    print("All unit tests for Setting Object Initialization has been passed.")
    return 0


if __name__ == "__main__":
    unit_test_settings()