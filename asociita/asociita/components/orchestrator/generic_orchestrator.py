import datasets, copy
from typing import Any, Union
from asociita.components.nodes.federated_node import FederatedNode
from asociita.models.pytorch.federated_model import FederatedModel
from asociita.utils.computations import Aggregators
from asociita.utils.handlers import Handler
from asociita.utils.loggers import Loggers
from asociita.utils.orchestrations import prepare_nodes, create_nodes, check_health, sample_nodes, train_nodes
from asociita.components.archiver.archive_manager import Archive_Manager
from asociita.components.settings.settings import Settings
from multiprocessing import Pool, Manager
from torch import nn


orchestrator_logger = Loggers.orchestrator_logger()
from multiprocessing import set_start_method
set_start_method("spawn", force=True)

class Orchestrator():
    def __init__(self, 
                 settings: Settings) -> None:
        """Orchestrator is a central object necessary for performing the simulation.
        It connects the nodes, maintain the knowledge about their state and manages the
        multithread pool. generic_orchestrator.orchestrator is a parent to all more
        specific orchestrators.
        
        -------------
        Args
            settings (settings): Settings object cotaining all the settings of the orchestrator.
       -------------
         Returns
            None"""
        self.settings = settings
        self.model = None
    
    
    def prepare_orchestrator(self, 
                             model: Any,
                             validation_data: datasets.arrow_dataset.Dataset,
                             ) -> None:
        """Loads the global model and validation data that will be used by the Orchestrator.
        
        -------------
        Args
        validation_data (datasets.arrow_dataset.Dataset): 
            Validation dataset that will be used by the Orchestrator.
        model (Any): Compiled or pre-compiled model that will 
            be used by the instance of the class.
        -------------
        Returns
            None"""
        self.validation_data = [validation_data]
        self.central_net = model
        self.central_model = FederatedModel(settings = self.settings.model_settings,
                                        net=model,
                                        local_dataset=self.validation_data,
                                        node_name='orchestrator')
    

    def model_initialization(self,
                             nodes_number: int,
                             model: Union[nn.Module, list[nn.Module]],
                             local_warm_start: bool = False,
                             ) -> list[nn.Module]:
        """Creates a list of neural nets (not FederatedModels!) that will be
        passed onto the nodes and converted into FederatedModels. If local_warm_start
        is set to True, the method call should be passed a list of models which
        length is equall to the number of nodes.
        
        -------------
        Args:
            nodes_number (int): number of nodes that will participate in the
                training.
            model (Union[nn.Module, list]): a neural net schematic (if warm
                start is set to False) or a number of different neural net schematics
                (if warm start is set to True) that will prepared for the nodes to be
                loaded as FederatedModels.
            local_warm_start (bool): A boolean value for switching on/off the warm start
                utility.
        -------------
        Returns:
            tuple(node_id: str, weights)
        """
        if local_warm_start == True:
            raise NotImplementedError("Local warm start is not implemented yet.")
        else:
            model_list = [copy.deepcopy(model) for _ in range(nodes_number)]
        return model_list


    def nodes_initialization(self,
                             nodes_list: list[FederatedNode],
                             model_list: list[nn.Module],
                             data_list: list[datasets.arrow_dataset.Dataset, 
                                    datasets.arrow_dataset.Dataset],
                             nodes_number: int) -> list[FederatedNode]:
        """Prepare instances of a FederatedNode object for a participation in 
        the Federated Training.  Contrary to the  create nodes function, 
        it accepts only already initialized instances of the FederatedNode
        object.
        
        -------------
        Args:
            nodess_list (list[FederatedNode]): list containing all the initialized
                FederatedNode instances.
            model_list (list[nn.Module]): list containing all the initialized 
                nn.Module objects. Note that conversion from nn.Module into the
                FederatedModel will occur at the local node level.
            data_list (list[..., ....]): list containing train set and test set
                wrapped in a hugging facr arrow_dataset.Dataset containers.
            nodes_number (int): Number of nodes that will participate in the training.
        -------------
        Returns:
            list[FederatedNode]"""
        
        results = []
        for node, model, dataset in zip(nodes_list, model_list, data_list):
            node.prepare_node(model, dataset)
            results.append(node)
        nodes_green = []
        for result in results:
            if check_health(result,
                            orchestrator_logger=orchestrator_logger):
                nodes_green.append(result)
        return nodes_green # Returning initialized nodes


    def train_protocol(self,
                nodes_data: list[datasets.arrow_dataset.Dataset, 
                datasets.arrow_dataset.Dataset]) -> None:
        """Performs a full federated training according to the initialized
        settings. The train_protocol of the generic_orchestrator.Orchestrator
        follows a classic FedAvg algorithm - it averages the local weights
        and aggregates them taking a weighted average.
        SOURCE: Communication-Efficient Learning of Deep Networks from Decentralized Data, H.B. McMahan et al.

        -------------
        Args:
            nodes_data(list[..., ....]): list containing train set and test set
                wrapped in a hugging facr arrow_dataset.Dataset containers.
        -------------
        Returns:
            None"""
        
        # SETTINGS -> CHANGE IF NEEDED
        iterations = self.settings.iterations
        nodes_number = self.settings.number_of_nodes
        local_warm_start = self.settings.local_warm_start
        nodes = [node for node in range(nodes_number)]
        sample_size = self.settings.sample_size
        if self.settings.enable_archiver == True:
            archive_manager = Archive_Manager(
                archive_manager = self.settings.archiver_settings,
                logger = orchestrator_logger)

        # CREATING FEDERATED NODES
        nodes_green = create_nodes(nodes, 
                                   self.settings.nodes_settings)


        # CREATING LOCAL MODELS (that will be loaded onto nodes)
        model_list = self.model_initialization(nodes_number=nodes_number,
                                               model=self.central_net)
        
        # INITIALIZING ALL THE NODES
        nodes_green = self.nodes_initialization(nodes_list=nodes_green,
                                                model_list=model_list,
                                                data_list=nodes_data,
                                                nodes_number=nodes_number)
            
        # TRAINING PHASE ----- FEDAVG
        with Manager() as manager:
            queue = manager.Queue() # creates a shared queue
            # create the pool of workers
            with Pool(sample_size) as pool: 
                for iteration in range(iterations):
                    orchestrator_logger.info(f"Iteration {iteration}")
                    weights = {}
                    # Sampling nodes and asynchronously apply the function
                    sampled_nodes = sample_nodes(nodes_green, 
                                                 sample_size=sample_size, 
                                                 orchestrator_logger=orchestrator_logger) # SAMPLING FUNCTION -> CHANGE IF NEEDED
                    results = [pool.apply_async(train_nodes, (node,)) for node in sampled_nodes]
                    # consume the results
                    for result in results:
                        node_id, model_weights = result.get()
                        weights[node_id] = copy.deepcopy(model_weights)
                    # Computing the average
                    avg = Aggregators.compute_average(weights) # AGGREGATING FUNCTION -> CHANGE IF NEEDED
                    # Updating the nodes
                    for node in nodes_green:
                        node.model.update_weights(avg)
                    # Upadting the orchestrator
                    self.central_model.update_weights(avg)

                    if self.settings.enable_archiver == True:
                        archive_manager.archive_training_results(iteration = iteration,
                                                                central_model=self.central_model,
                                                                nodes=nodes_green)

        orchestrator_logger.critical("Training complete")
        return 0
                    
