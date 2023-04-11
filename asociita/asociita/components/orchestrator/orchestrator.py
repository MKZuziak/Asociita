import types, logging, datasets, multiprocess, copy
from typing import Any
from asociita.components.nodes.federated_node import FederatedNode
from multiprocessing import Pool, Manager


def prepare_nodes(node: FederatedNode, 
                 model: Any,
                 dataset: list[datasets.arrow_dataset.Dataset, 
                               datasets.arrow_dataset.Dataset],
                               query) -> str:
    """Used to connect the node and prepare it for training.
    Updates instances of a FederatedNode object and
    puts it into communication_queue.
    
    -------------
    Args:
        node (int): ID of the node that we want to connect.
        model (Any): Compiled or pre-compiled model to be trained.
        dataset (list[datasets.arrow_dataset.Dataset, 
                datasets.arrow_dataset.Dataset]): A dataset in the
                format ["train_data", "test_data"] that will be used 
                by the selected node.
        comunication_queue (multiprocess.Manager.Queue): Communication queue.
    -------------
    Returns:
        message(str): "OK" """
    node.prepare_node(model=model, data=dataset)
    query.put(node)
    return "OK"


def create_nodes(node_id: int) -> list[FederatedNode]: 
    """Creates a list of nodes that will be connected to the 
    orchestrator and contained in a list[FederatedNode] container.
    -------------
    Args:
        node (int): ID of the node that we want to connect.
    -------------
    Returns:
        list[FederatedNode]: List of nodes that were created.
    """
    nodes = [FederatedNode(id) for id in node_id]
    return nodes


def check_health(node: FederatedNode) -> int:
    """Checks whether node has successfully conducted the transaction
    and can be moved to the next phase of the training. According to the
    adopted standard - if node.state == 0, node is ready for the next
    transaction. On the contrary, if node.state == 1, then node must be 
    excluded from the simulation (internal error)."""
    if node.state == 0:
        logging.warning(f"Node {node.node_id} was updated successfully.")
        return True
    else:
        logging.warning(f"Node {node.node_id} failed during the update.")
        return False


class Orchestrator():
    def __init__(self, settings: dict) -> None:
        """Orchestrator is a central object necessary for performing the simulation.
        It connects the nodes, maintain the knowledge about their state and manages the
        multithread pool.
        
        Parameters
        ----------
        settings : dict
            Dictionary object cotaining all the settings of the orchestrator.
        
        Returns
        ----------
        None"""
        self.state = 1 # Attribute controlling the state of the object.
                         # 0 - initialized, resting
                         # 1 - initialized, in run-time
        
        self.settings = settings # Settings attribute (dict)
        self.state = 0
    

    def load_model(self, model: Any) -> None:
        """Loads the global model that will be used as the orchestrator's main model
        In contrast to the client object, load_model and load_data are separated in the
        instance of the orchestrator class.
        
        Parameters
        ----------
        model : Any
            Compiled or pre-compiled model that will be used by the instance of the class.
        
        Returns
        ----------
        None"""
        assert self.state == 0, f"Object {self} is not resting, previous operation is still active."
        self.state = 1
        
        try:
            self.model = model
        except:
            logging.critical("Failed to load the model")
        
        if self.model != None:
            self.state = 0


    def load_data(self, validation_data: datasets.arrow_dataset.Dataset) -> None:
        """Loads the validation data that will be used by the Orchestrator.
        In contrast to the client object, load_model and load_data are separated 
        in the instance of the orchestrator class.
        
        Parameters
        ----------
        validation_data: datasets.arrow_dataset.Dataset
            Validation dataset that will be used by the Orchestrator.
        
        Returns
        ----------
        None"""
        assert self.state == 0, f"Object {self} is not resting, previous operation is still active."
        self.state = 1
        
        try:
            self.validation_data = validation_data
        except:
            logging.critical("Failed to load the data")
        
        if self.validation_data != None:
            self.state = 0
    

    def training_protocol(self,
                          nodes_data: list[datasets.arrow_dataset.Dataset, 
                               datasets.arrow_dataset.Dataset]):
        
        # Defining the settings
        iterations = self.settings['iterations']
        nodes_number = self.settings['number_of_nodes']
        local_warm_start = self.settings["local_warm_start"]
        nodes = self.settings["nodes"]
        

        # Creating a list containing nodes, i.e. FederatedNode objects.
        av_nodes = create_nodes(nodes)


        # Creating a multiprocess manager and communication queue.
        #manager = multiprocess.Manager()
        #communication_queue = manager.Queue()


        # Copying the the local model n times or initiating with local warm start.
        if local_warm_start == True:
            raise("Beginning the training with different local models not implemented yet.")
        else:
            model_list = [copy.deepcopy(self.model) for _ in range(nodes_number)]
        
        # create the manager
        with Manager() as manager:
            # create the shared queue
            queue = manager.Queue()
            # create the pool of workers
            with Pool(nodes_number) as pool:
                args = [(node, model, dataset) for 
                        node, model, dataset in zip(av_nodes, model_list, nodes_data)]
                results = [
                    pool.apply_async(prepare_nodes, (node, model, dataset, queue))
                    for node, model, dataset in zip(av_nodes, model_list, nodes_data)
                ]
                for result in results:
                    nodes_green = []
                    _ = result.get()
                    updated_node = queue.get()
                    if check_health(updated_node):
                        nodes_green.append(updated_node)
                    
