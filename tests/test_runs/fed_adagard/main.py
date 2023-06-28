from asociita.components.orchestrator.fedopt_orchestrator import Fedopt_Orchestrator
from asociita.components.settings.settings import Settings
from asociita.datasets.fetch_data import load_data
from asociita.models.pytorch.mnist import MNIST_CNN
import os

def main():
    config = {
        "orchestrator": {
            "iterations": 400,
            "number_of_nodes": 1000,
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
    config['orchestrator']['archiver']['metrics_savepath'] = os.getcwd()
    config['orchestrator']['archiver']['orchestrator_filename'] = 'test1.csv'
    config['orchestrator']['archiver']['clients_on_central_filename'] = 'test2.csv'
    config['orchestrator']['archiver']['central_on_local_filename'] = 'test3.csv'
    config['orchestrator']['archiver']['orchestrator_model_save_path'] = os.path.join(os.getcwd(), 'models')
    config['orchestrator']['archiver']['nodes_model_save_path'] = os.path.join(os.getcwd(), 'models')
    data_config = {
        "dataset_name" : "mnist",
        "split_type" : "homogeneous",
        "shards": 1000,
        "local_test_size": 0.2,
        "transformations": {},
        "imbalanced_clients": {},
        "save_dataset": False,
        "save_transformations": False,
        "save_blueprint": False,
        "agents": 1000}
    settings = Settings(initialization_method='dict',
                                dict_settings = config)
    data = load_data(data_config)
    # DATA: Selecting data for the orchestrator
    orchestrator_data = data[0]
    # DATA: Selecting data for nodes
    nodes_data = data[1]
    model = MNIST_CNN()
        
    gen_orch = Fedopt_Orchestrator(settings)
    
    gen_orch.prepare_orchestrator(
            model=model, 
            validation_data=orchestrator_data)
    
    signal = gen_orch.train_protocol(nodes_data=nodes_data)


if __name__ == "__main__":
    main()