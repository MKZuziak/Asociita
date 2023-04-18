from asociita.components.orchestrator.orchestrator import Orchestrator
from asociita.components.nodes.federated_node import FederatedNode
from asociita.models.pytorch.mnist import MnistNet
from asociita.datasets.fetch_data import load_data
from asociita.utils.helpers import Helpers
import os

if __name__ == "__main__":
    # CONFIGURATION: Training configuration
    save_path = os.path.join(os.getcwd(), r'examples', r'metrics_fedopt.csv')
    settings = Helpers.load_from_json(os.path.join(os.getcwd(), 'examples', 'simulation_configuration.json'))
    settings["orchestrator"]['save_path'] = os.path.join(os.getcwd(), 'examples', 'metrics.csv')
    # CONFIGURATION: Dataset configuration
    data_settings = {
            "dataset_name" : 'mnist',
            "split_type" : 'random_uniform',
            "shards": 10,
            "local_test_size": 0.2}
    

    # DATA: Loading the data
    data = load_data(data_settings)
    # DATA: Selecting data for the orchestrator
    orchestrator_data = data[0]
    # DATA: Selecting data for nodes
    nodes_data = data[1]


    # MODEL: Using utils to retrieve a model
    model = MnistNet()
    
    
    # SIMULATION: Creating an Orchestrator object
    orchestrator = Orchestrator(settings=settings)
    # SIMULATION: Loading the model onto the orchestrator
    orchestrator.prepare_orchestrator(model=model, validation_data=orchestrator_data)


    # TRAINING PHASE: running the training protocol.
    #orchestrator.fed_avg(nodes_data=nodes_data)
    orchestrator.fed_opt(nodes_data=nodes_data)

    # Checking the model
    #model = orchestrator.central_model
    #print(model)
    