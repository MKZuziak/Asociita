from asociita.components.orchestrator.generic_orchestrator import Orchestrator
from asociita.components.orchestrator.fedopt_orchestrator import Fedopt_Orchestrator
from asociita.components.orchestrator.evaluator_orchestrator import Evaluator_Orchestrator
from asociita.components.nodes.federated_node import FederatedNode
from asociita.models.pytorch.fmnist import create_FashionMnistNet
from asociita.models.pytorch.mnist import MnistNet
from asociita.models.pytorch.cifar10 import CifarNet
from asociita.datasets.fetch_data import load_data
from asociita.utils.helpers import Helpers
import os
import time
import timm

if __name__ == "__main__":
    # CONFIGURATION: Training configuration
    save_path = os.path.join(os.getcwd(), 'examples')
    settings = Helpers.load_from_json(os.path.join(os.getcwd(), 'examples',  'simulation_configurations', 'Contribution_example.json'))
    data_settings = Helpers.load_from_json(os.path.join(os.getcwd(), 'examples', 'dataset_configurations', 'Heterogeneous_Size.json'), convert_keys=True)
    results_save_path = os.path.join(save_path, "results")
    settings["orchestrator"]["metrics_save_path"] = results_save_path
    settings["orchestrator"]["archiver"]["metrics_savepath"] = results_save_path
    settings["orchestrator"]["archiver"]["orchestrator_filename"] = "FMNIST_FEDOPT_YOGI.csv"
    settings["orchestrator"]["archiver"]["central_on_local_filename"] = "central_on_local.csv"
    settings["orchestrator"]["archiver"]["orchestrator_model_save_path"] = results_save_path
    settings["orchestrator"]["archiver"]["nodes_model_save_path"] = results_save_path
    data_settings['save_path'] = os.path.join(os.getcwd(), 'examples', 'datasets')
    # DATA: Loading the data
    data = load_data(data_settings)
    # DATA: Selecting data for the orchestrator
    orchestrator_data = data[0]
    # DATA: Selecting data for nodes
    nodes_data = data[1]


    # MODEL: Using utils to retrieve a model
    model = MnistNet()
    # SIMULATION: Creating an Orchestrator object
    orchestrator = Fedopt_Orchestrator(settings=settings)
    # SIMULATION: Loading the model onto the orchestrator
    orchestrator.prepare_orchestrator(model=model, validation_data=orchestrator_data)

    # TRAINING PHASE: running the training protocol.
    #orchestrator.fed_avg(nodes_data=nodes_data)
    orchestrator.train_protocol(nodes_data=nodes_data)