from asociita.components.orchestrator.fedopt_orchestrator import Fedopt_Orchestrator
from asociita.components.settings.settings import Settings
from asociita.datasets.fetch_data import load_data
from asociita.models.pytorch.mnist import MNIST_CNN
import unittest
import os

class FedoptOrchestratorInitCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "orchestrator": {
                "iterations": 2000,
                "number_of_nodes": 800,
                "local_warm_start": False,
                "sample_size": 50,
                "metrics_save_path": "None",
                'enable_archiver': False,
                'enable_optimizer': True,
                'enable_evaluator': False,
                "optimizer": {
                    "name": "FedAdagard",
                    "learning_rate": 0.03162277660168379,
                    "b1": 0.3,
                    "tau": 0.01}
            },
            "nodes":{
            "local_epochs": 2,
            "model_settings": {
                "optimizer": "RMS",
                "batch_size": 32,
                "learning_rate": 0.0031622776601683794}
                }
        }
        self.data_config = {
            "dataset_name" : "mnist",
            "split_type" : "heterogeneous_size",
            "shards": 800,
            "local_test_size": 0.2,
            "transformations": {},
            "imbalanced_clients": {},
            "save_dataset": False,
            "save_transformations": False,
            "save_blueprint": False,
            "agents": 800}
        self.settings = Settings(initialization_method='dict',
                                 dict_settings = self.config)
        self.data = load_data(self.data_config)
        # DATA: Selecting data for the orchestrator
        self.orchestrator_data = self.data[0]
        # DATA: Selecting data for nodes
        self.nodes_data = self.data[1]
        self.model = MNIST_CNN()
        
    def testInit(self) -> None:
        self.gen_orch = Fedopt_Orchestrator(self.settings, full_debug = True, batch_job = False, batch=15)
    
    def testPreparationPhase(self) -> None:
        self.gen_orch.prepare_orchestrator(
            model=self.model, 
            validation_data=self.orchestrator_data)
    
    def testTrainingPhase(self) -> None:
        signal = self.gen_orch.train_protocol(nodes_data=self.nodes_data)
        self.assertEqual(signal, 0)

def unit_test_fedoptorchestrator():
    case = FedoptOrchestratorInitCase()
    case.setUp()
    case.testInit()
    case.testPreparationPhase()
    case.testTrainingPhase()
    print("All unit test for Generic Object Orchestrator were passed")


class FedoptOrchestratorInitCase_Warchiver(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
        "orchestrator": {
            "iterations": 2000,
            "number_of_nodes": 800,
            "local_warm_start": False,
            "sample_size": 50,
            'enable_archiver': True,
            'enable_optimizer': True,
            "enable_evaluator": False,
            "archiver":{
                "orchestrator": True,
                "clients_on_central": True,
                "central_on_local": True,
                "log_results": True,
                "save_results": True,
                "save_orchestrator_model": True,
                "save_nodes_model": True,
                "metrics_savepath": "None",
                "orchestrator_filename": "None",
                "clients_on_central_filename": "None",
                "central_on_local_filename": "None",
                "orchestrator_model_save_path": "None",
                "nodes_model_save_path": "None"},
                "optimizer": {
                    "name": "FedAdagard",
                    "learning_rate": 0.01,
                    "b1": 0.3,
                    "tau": 0.01}
        },
        "nodes":{
        "local_epochs": 2,
        "model_settings": {
            "optimizer": "RMS",
            "batch_size": 32,
            "learning_rate": 0.001}
            }}
        self.config['orchestrator']['archiver']['metrics_savepath'] = os.getcwd()
        self.config['orchestrator']['archiver']['orchestrator_filename'] = 'test1.csv'
        self.config['orchestrator']['archiver']['clients_on_central_filename'] = 'test2.csv'
        self.config['orchestrator']['archiver']['central_on_local_filename'] = 'test3.csv'
        self.config['orchestrator']['archiver']['orchestrator_model_save_path'] = os.getcwd()
        self.config['orchestrator']['archiver']['nodes_model_save_path'] = os.getcwd()
        self.data_config = {
            "dataset_name" : "mnist",
            "split_type" : "homogeneous",
            "shards": 800,
            "local_test_size": 0.2,
            "transformations": {},
            "imbalanced_clients": {},
            "save_dataset": False,
            "save_transformations": False,
            "save_blueprint": False,
            "agents": 800}
        self.settings = Settings(initialization_method='dict',
                                 dict_settings = self.config)
        self.data = load_data(self.data_config)
        # DATA: Selecting data for the orchestrator
        self.orchestrator_data = self.data[0]
        # DATA: Selecting data for nodes
        self.nodes_data = self.data[1]
        self.model = MNIST_CNN()
        
    def testInit(self) -> None:
        self.gen_orch = Fedopt_Orchestrator(self.settings)
    
    def testPreparationPhase(self) -> None:
        self.gen_orch.prepare_orchestrator(
            model=self.model, 
            validation_data=self.orchestrator_data)
    
    def testTrainingPhase(self) -> None:
        signal = self.gen_orch.train_protocol(nodes_data=self.nodes_data)
        self.assertEqual(signal, 0)

def unit_test_fedoptorchestrator_warchiver():
    case = FedoptOrchestratorInitCase_Warchiver()
    case.setUp()
    case.testInit()
    case.testPreparationPhase()
    case.testTrainingPhase()
    print("All unit test for Generic Object Orchestrator with supporting archiver were passed")

if __name__ == "__main__":
    unit_test_fedoptorchestrator_warchiver()