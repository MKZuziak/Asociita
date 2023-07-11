from asociita.components.settings.settings import Settings
from asociita.components.settings.init_settings import init_settings
import unittest
import time

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


class ArchiverInitTestCase(SettingsInitTestCase):
    def setUp(self) -> None:
        self.config = {
            "orchestrator": {
                "iterations": 10,
                "number_of_nodes": 5,
                "local_warm_start": False,
                "sample_size": 2,
                "metrics_save_path": "None",
                'enable_archiver': True,
                "archiver": {
                    "evaluate_orchestrator": True,
                    "clients_on_central" : True,
                    "central_on_local": True,
                    "log_results": True,
                    "save_results": True,
                    "save_orchestrator_model": True,
                    "save_nodes_model": True,
                    "metrics_savepath": None,
                    "orchestrator_model_savepath": None,
                    "nodes_model_savepath": None,
                    "orchestrator_filename": None,
                    "clients_on_central_filename": None,
                    "central_on_local_filename": None
                }
            },
            "nodes":{
            "local_epochs": 3,
            "model_settings": {
                "optimizer": "RMS",
                "batch_size": 64,
                "learning_rate": 0.0031622776601683794}
                }
        }
    
    def testPropertyRetrive(self) -> None:
        self.assertIsNotNone(self.settings.orchestrator_settings['archiver'])
        self.assertEqual(self.settings.orchestrator_settings['archiver']['evaluate_orchestrator'], True)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['clients_on_central'], True)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['central_on_local'], True)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['log_results'], True)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['save_orchestrator_model'], True)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['save_nodes_model'], True)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['evaluate_orchestrator'], True)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['metrics_savepath'], None)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['orchestrator_model_savepath'], None)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['nodes_model_savepath'], None)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['orchestrator_filename'], None)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['clients_on_central_filename'], None)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['central_on_local_filename'], None)


class DefaultArchiverInitTest(ArchiverInitTestCase):
    def setUp(self):
        self.config = {"orchestrator": {"enable_archiver": True}}
    
    def testPropertyRetrive(self) -> None:
        self.assertIsNotNone(self.settings.orchestrator_settings['archiver'])
        self.assertEqual(self.settings.orchestrator_settings['archiver']['orchestrator'], True)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['clients_on_central'], False)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['central_on_local'], False)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['log_results'], True)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['save_orchestrator_model'], False)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['save_nodes_model'], False)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['nodes_model_savepath'], None)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['orchestrator_filename'], "orchestrator_results.csv")
        self.assertEqual(self.settings.orchestrator_settings['archiver']['clients_on_central_filename'], None)
        self.assertEqual(self.settings.orchestrator_settings['archiver']['central_on_local_filename'], None)

class FormArchiverInitTest(ArchiverInitTestCase):
    def setUp(self) -> None:
        self.config = {
            "orchestrator": {
                "iterations": 10,
                "number_of_nodes": 5,
                "local_warm_start": False,
                "sample_size": 2,
                "metrics_save_path": "None",
                'enable_archiver': True,
                "archiver": {
                    "evaluate_orchestrator": True,
                    "clients_on_central" : True,
                    "central_on_local": True,
                    "log_results": True,
                    "save_results": True,
                    "save_orchestrator_model": True,
                    "save_nodes_model": True,
                    "metrics_savepath": None,
                    "orchestrator_model_savepath": None,
                    "nodes_model_savepath": None,
                    "orchestrator_filename": None,
                    "clients_on_central_filename": None,
                    "central_on_local_filename": None,
                    "form_archive": True
                }
            },
            "nodes":{
            "local_epochs": 3,
            "model_settings": {
                "optimizer": "RMS",
                "batch_size": 64,
                "learning_rate": 0.0031622776601683794}
                }
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

    case4 = ArchiverInitTestCase()
    case4.setUp()
    case4.testInit()
    case4.testPropertyRetrive()

    case5 = DefaultArchiverInitTest()
    case5.setUp()
    case5.testInit()
    case5.testPropertyRetrive()

    time.sleep(2) #Nec. to avoid directory overlap
    case6 = FormArchiverInitTest()
    case6.setUp()
    case6.testInit()
    print("All unit tests for Setting Object Initialization has been passed.")
    return 0


if __name__ == "__main__":
    unit_test_settings()