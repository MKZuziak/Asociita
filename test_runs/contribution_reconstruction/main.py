from asociita.components.orchestrator.evaluator_orchestrator import Evaluator_Orchestrator
from asociita.components.nodes.federated_node import FederatedNode
from asociita.models.pytorch.fmnist import create_FashionMnistNet
from asociita.models.pytorch.mnist import MnistNet
from asociita.datasets.fetch_data import load_data
from asociita.utils.helpers import Helpers
import os
import csv
import copy

if __name__ == "__main__":
    # CONFIGURATION: Training configuration
    save_path = os.path.join(os.getcwd(), 'test_runs', "contribution_reconstruction")
    settings = Helpers.load_from_json(os.path.join(os.getcwd(), 'test_runs',  'contribution_reconstruction', 'simulation_configurations', 'simulation_configuration.json'))
    data_settings = Helpers.load_from_json(os.path.join(os.getcwd(), 'test_runs', 'contribution_reconstruction', 'dataset_configurations', "dataset_configuration.json"), convert_keys=True)
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
    orchestrator_data_copy = copy.deepcopy(orchestrator_data)
    # DATA: Selecting data for nodes
    nodes_data = data[1]
    nodes_data_copy = copy.deepcopy(nodes_data)

    # BASELINE CONFIGURATION AND TEST RUN
    model = MnistNet()
    orchestrator = Evaluator_Orchestrator(settings=settings)
    orchestrator.prepare_orchestrator(model=model, validation_data=orchestrator_data_copy)
    orchestrator.train_protocol(nodes_data=nodes_data_copy)

    # DIRECT LOO RECONSTRUCTION - FOR RESULT CHECK
    nodes = settings['orchestrator']['nodes']
    # BASELINE SCORE
    model = MnistNet()
    global_result = orchestrator.central_model.evaluate_model()[1]
    path = os.path.join(results_save_path, "real_results.csv")
    loo_results = {}
    
    for node in nodes:
        orchestrator_data_copy = copy.deepcopy(orchestrator_data)
        nodes_data_copy = copy.deepcopy(nodes_data)
        del nodes_data_copy[node]
        settings['orchestrator']['evaluation']['LOO_OR'] = False
        settings['orchestrator']["nodes"] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        settings['orchestrator']["number_of_nodes"] = 9
        model = MnistNet()
        orchestrator = Evaluator_Orchestrator(settings=settings)
        orchestrator.prepare_orchestrator(model=model, validation_data=orchestrator_data_copy)
        orchestrator.train_protocol(nodes_data=nodes_data_copy)

        local_result = orchestrator.central_model.evaluate_model()[1]
        loo = global_result - local_result
        loo_results[node] = loo

    
    with open(path, 'a+', newline='') as saved_file:
        for node, value in loo_results.items():
            saved_file.write(f"{node},{value}")
            saved_file.write("\n")
