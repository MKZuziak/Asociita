import types, logging, datasets, multiprocess, copy, random
from typing import Any, Tuple, List, Dict, AnyStr
from asociita.components.nodes.federated_node import FederatedNode
from asociita.models.pytorch.federated_model import FederatedModel
from multiprocessing import Pool, Manager


def prepare_nodes(node: FederatedNode, 
                 model: Any,
                 dataset: list[datasets.arrow_dataset.Dataset, 
                               datasets.arrow_dataset.Dataset],
                               query) -> AnyStr:
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


def create_nodes(node_id: int, nodes_settings) -> list[FederatedNode]: 
    """Creates a list of nodes that will be connected to the 
    orchestrator and contained in a list[FederatedNode] container.
    -------------
    Args:
        node (int): ID of the node that we want to connect.
    -------------
    Returns:
        list[FederatedNode]: List of nodes that were created.
    """
    nodes = [FederatedNode(id, nodes_settings) for id in node_id]
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


def sample_nodes(nodes: list[FederatedNode], sample_size: int) -> list[FederatedNode]:
    """Sample the nodes given the provided sample size. If sample_size is bigger
    or equal to the number of av. nodes, the sampler will return the original list.
     -------------
    Args:
        nodes (list[FederatedNode]): original list of nodes to be sampled from
        sample_size (int): size of the sample.
    -------------
    Returns:
        list[FederatedNode]: List of sampled nodes."""
    if len(nodes) <= sample_size:
        logging.warning("Sample size should be smaller than the size of the population, returning the original list")
        return nodes
    else:
        sample = random.sample(nodes, sample_size)
        return sample


def train_nodes(node: FederatedNode) -> tuple[int, tuple[List[float], List[float], List[float]]]:
    """Used to command the node to start the local training.
    Invokes .train_local_model method and returns the results.
    -------------
    Args:
        node (FederatedNode object): Node that we want to train.
    -------------
    Returns:
        tuple(node_id: str, weights)"""
    node_id, weights = node.train_local_model()
    return (node_id, weights)


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
        self.model = None
        self.state = 0
    

    def prepare_orchestrator(self, 
                             model: Any,
                             validation_data: datasets.arrow_dataset.Dataset,
                             ) -> None:
        """Loads the global model and validation data that will be used by the Orchestrator.
        In contrast to the client object, load_model and load_data are separated 
        in the instance of the orchestrator class.
        
        Parameters
        ----------
        validation_data (datasets.arrow_dataset.Dataset): 
            Validation dataset that will be used by the Orchestrator.
        model (Any): Compiled or pre-compiled model that will 
            be used by the instance of the class.
        Returns
        ----------
        None"""
        assert self.state == 0, f"Object {self} is not resting, previous operation is still active."
        self.state = 1
        self.validation_data = [validation_data]
        self.central_net = model
        self.central_model = FederatedModel(settings = self.settings["nodes_settings"]["model_settings"],
                                        net=model,
                                        local_dataset=self.validation_data,
                                        node_name='orchestrator')
        if self.central_model != None and self.central_net != None:
            self.state = 0


    def training_protocol(self,
                          nodes_data: list[datasets.arrow_dataset.Dataset, 
                               datasets.arrow_dataset.Dataset]):
        
        # Defining the settings
        iterations = self.settings['iterations']
        nodes_number = self.settings['number_of_nodes']
        local_warm_start = self.settings["local_warm_start"]
        nodes = self.settings["nodes"]
        sample_size = self.settings["sample_size"]
        nodes_settings = self.settings["nodes_settings"]
        

        # Creating a list containing nodes, i.e. FederatedNode objects.
        nodes_green = create_nodes(nodes, nodes_settings)


        # Copying the the local model n times or initiating with local warm start.
        if local_warm_start == True:
            raise("Beginning the training with different local models not implemented yet.")
        else:
            model_list = [copy.deepcopy(self.central_net) for _ in range(nodes_number)]
        
        # PHASE ONE: NODES INITIALIZATION
        # create the manager
        with Manager() as manager:
            # create the shared queue
            queue = manager.Queue()
            
            
            # create the pool of workers
            with Pool(nodes_number) as pool:
                # asynchronously apply the function
                results = [
                    pool.apply_async(prepare_nodes, (node, model, dataset, queue))
                    for node, model, dataset in zip(nodes_green, model_list, nodes_data)
                ]
                # consume the results
                # Define a list of healthy nodes
                nodes_green = []
                for result in results:
                    # query for results
                    _ = result.get()
                    updated_node = queue.get()
                    # Adds to list only if the node is healthy
                    if check_health(updated_node):
                        nodes_green.append(updated_node)
            
        #PHASE TWO: MODEL TRAINING
        with Manager() as manager:
            # create the shared queue
            queue = manager.Queue()

            # create the pool of workers
            with Pool(sample_size) as pool:
                for iteration in range(iterations):
                    logging.info(f"Iteration {iteration}")
                    weights = {}
                    
                    # Sampling nodes and asynchronously apply the function
                    sampled_nodes = sample_nodes(nodes_green, sample_size=sample_size)
                    results = [pool.apply_async(train_nodes, (node,)) for node in sampled_nodes] # TODO: TRAIN NODES
                    # consume the results
                    for result in results:
                        node_id, model_weights = result.get()
                        weights[node_id] = copy.deepcopy(model_weights)
                    
                    # Computing the average
                    avg = Utils.compute_average(weights) #TODO: UTILS, COMPUTE_AVERAGE
                    # Updating the nodes
                    for node in nodes_green:
                        node.model.update_weights(avg)
                    # Upadting the orchestrator
                    self.model.update_weights(avg) #TODO
        
        logging.info("Training complete")
                    
