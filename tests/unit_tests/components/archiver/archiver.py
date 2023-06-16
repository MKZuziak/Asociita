from asociita.components.archiver.archive_manager import Archive_Manager
import unittest

class ArchiverInitCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "orchestrator": True,
            "clients_on_central": True,
            "central_on_local": True,
            "log_results": True,
            "save_results": True,
            "save_orchestrator_model": True,
            "save_nodes_model": False,
            "metrics_savepath": "None",
            "orchestrator_filename": "None",
            "clients_on_central_filename": "None",
            "central_on_local_filename": "None",
            "orchestrator_model_save_path": "None",
            "nodes_model_save_path": "None"}
    
    def testArchiverInit(self):
        Archive_Manager(self.config)


def unit_test_archiver():
    case = ArchiverInitCase()
    case.setUp()
    case.testArchiverInit()
    print("All unit test for Archiver Initialization Case were passed")

unit_test_archiver()