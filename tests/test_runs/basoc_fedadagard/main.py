from asociita.components.orchestrator.evaluator_orchestrator import Evaluator_Orchestrator
from asociita.datasets.fetch_data import load_data
from asociita.models.pytorch.mnist import MNIST_CNN
import os
from asociita.components.settings.init_settings import init_settings

def main():
    cwd = os.getcwd()
    config = {
        "orchestrator": {
            "iterations": 10,
            "number_of_nodes": 10,
            "sample_size": 5,
            'enable_archiver': True,
            "archiver":{
                "root_path": cwd,
                "orchestrator": True,
                "clients_on_central": True,
                "central_on_local": True,
                "log_results": True,
                "save_results": True,
                "save_orchestrator_model": True,
                "save_nodes_model": True,
                "form_archive": True
                },
            "optimizer": {
                "name": "FedAdagard",
                "learning_rate": 0.1,
                "b1": 0.1,
                "tau": 0.1},
            "evaluator" : {
            "LOO_OR": False,
            "Shapley_OR": False,
            "IN_SAMPLE_LOO": True,
            "IN_SAMPLE_SHAP": False,
            "preserve_evaluation": {
                "preserve_partial_results": True,
                "preserve_final_results": True
            },
            "full_debug": True,
            "number_of_workers": 50}
        },
        "nodes":{
        "local_epochs": 2,
        "model_settings": {
            "optimizer": "RMS",
            "batch_size": 32,
            "learning_rate": 0.001}
    }}
    data_config = {
        "dataset_name" : "mnist",
        "split_type" : "homogeneous",
        "shards": 10,
        "local_test_size": 0.2,
        "transformations": {},
        "imbalanced_clients": {},
        "save_dataset": False,
        "save_transformations": False,
        "save_blueprint": False,
        "agents": 10}
    settings = init_settings(orchestrator_type='evaluator',
                             allow_default=True,
                             initialization_method='dict',
                             dict_settings=config)
    data = load_data(data_config)
    # DATA: Selecting data for the orchestrator
    orchestrator_data = data[0]
    # DATA: Selecting data for nodes
    nodes_data = data[1]
    model = MNIST_CNN()
        
    gen_orch = Evaluator_Orchestrator(settings)
    
    gen_orch.prepare_orchestrator(
            model=model, 
            validation_data=orchestrator_data)
    
    signal = gen_orch.train_protocol(nodes_data=nodes_data)


if __name__ == "__main__":
    main()
