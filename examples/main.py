from asociita.components.orchestrator.generic_orchestrator import Orchestrator
from asociita.components.orchestrator.fedopt_orchestrator import Fedopt_Orchestrator
from asociita.components.orchestrator.evaluator_orchestrator import Evaluator_Orchestrator
from asociita.components.nodes.federated_node import FederatedNode
from asociita.models.pytorch.mnist import MnistNet
from asociita.datasets.fetch_data import load_data
from asociita.utils.helpers import Helpers
import os
import time

if __name__ == "__main__":
    # CONFIGURATION: Training configuration
    save_path = os.path.join(os.getcwd(), r'examples')
    settings = Helpers.load_from_json(os.path.join(os.getcwd(), 'examples', 'Contribution_example.json'))
    settings["orchestrator"]['save_path'] = save_path
    # CONFIGURATION: Dataset configuration
    data_settings = {
            "dataset_name" : 'mnist',
            "split_type" : 'same_dataset',
            "shards": 2,
            "local_test_size": 0.2,
            "agents": 4}
    

    # DATA: Loading the data
    data = load_data(data_settings)
    # DATA: Selecting data for the orchestrator
    orchestrator_data = data[0]
    # DATA: Selecting data for nodes
    nodes_data = data[1]


    # MODEL: Using utils to retrieve a model
    model = MnistNet()
    
    st = time.time()
    # SIMULATION: Creating an Orchestrator object
    orchestrator = Evaluator_Orchestrator(settings=settings)
    # SIMULATION: Loading the model onto the orchestrator
    orchestrator.prepare_orchestrator(model=model, validation_data=orchestrator_data)


    # TRAINING PHASE: running the training protocol.
    #orchestrator.fed_avg(nodes_data=nodes_data)
    orchestrator.train_protocol(nodes_data=nodes_data)
    et = time.time()
    elapsed_time = et - st
    print(f"Execution time: {elapsed_time}, seconds")

    # Checking the model
    #model = orchestrator.central_model
    #print(model)
    