import datasets 
import copy
from typing import Union
from asociita.components.nodes.federated_node import FederatedNode
from asociita.models.pytorch.federated_model import FederatedModel
from asociita.utils.computations import Aggregators
from asociita.utils.loggers import Loggers
from asociita.utils.orchestrations import create_nodes, check_health, sample_nodes, train_nodes
from asociita.components.archiver.archive_manager import Archive_Manager
from asociita.components.settings.settings import Settings
from multiprocessing import Pool
from torch import nn
from asociita.utils.debugger import log_gpu_memory
from asociita.utils.helpers import Helpers

# set_start_method set to 'spawn' to ensure compatibility across platforms.
orchestrator_logger = Loggers.orchestrator_logger()
from multiprocessing import set_start_method
set_start_method("spawn", force=True)


class Orchestrator():
    """Orchestrator is a central object necessary for performing the simulation.
        It connects the nodes, maintain the knowledge about their state and manages the
        multithread pool. generic_orchestrator.orchestrator is a parent to all more
        specific orchestrators."""
    
    
    def __init__(self, 
                 settings: Settings,
                 **kwargs) -> None:
        """Orchestrator is initialized by passing an instance
        of the Settings object. Settings object contains all the relevant configurational
        settings that an instance of the Orchestrator object may need to complete the simulation.
        
        Parameters
        ----------
        settings : Settings
            An instance of the settings object cotaining all the settings 
            of the orchestrator.
        **kwargs : dict, optional
            Extra arguments to enable selected features of the Orchestrator.
            passing full_debug to **kwargs, allow to enter a full debug mode.
       
       Returns
       -------
       None
        """
        self.settings = settings
        self.model = None
        # Special option to enter a full debug mode.
        if kwargs.get("full_debug"):
            self.full_debug = True
        else:
            self.full_debug = False
        if kwargs.get("batch_job"):
            self.batch_job = True
            self.batch = kwargs["batch"]
        else:
            self.batch_job = False
    
    
    def prepare_orchestrator(self, 
                             model: nn,
                             validation_data: datasets.arrow_dataset.Dataset,
                             ) -> None:
        """Loads the orchestrator's test data and creates an instance
        of the Federated Model object that will be used throughout the training.
        
        Parameters
        ----------
        validation_data : datasets.arrow_dataset.Dataset:
            Validation dataset that will be used by the Orchestrator.
        model : torch.nn
            Model architecture that will be used throughout the training.
        
        Returns
        -------
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
        
        Parameters
        ----------
        nodes_number: int 
            number of nodes that will participate in the training.
        model: Union[nn.Module, list[nn.Module]] 
            a neural net schematic (if warm start is set to False) or 
            a number of different neural net schematics
            (if warm start is set to True) that 
            are prepared for the nodes to be loaded as FederatedModels.
        local_warm_start: bool, default False
            boolean value for switching on/off the warm start utility.
        
        Returns
        -------
        list[nn.Module]
            returns a list containing an instances of torch.nn.Module class.
        
        Raises
        ------
        NotImplemenetedError
            If local_warm_start is set to True.
        """
        if local_warm_start == True:
            raise NotImplementedError("Local warm start is not implemented yet.")
        else:
            # Deep copy is nec. because the models will have different (non-shared) parameters
            model_list = [copy.deepcopy(model) for _ in range(nodes_number)]
        return model_list


    def nodes_initialization(self,
                             nodes_list: list[FederatedNode],
                             model_list: list[nn.Module],
                             data_list: list[datasets.arrow_dataset.Dataset, 
                                    datasets.arrow_dataset.Dataset]
                                    ) -> list[FederatedNode]:
        """Prepare instances of a FederatedNode object for a participation in 
        the Federated Training.  Contrary to the 'create nodes' function, 
        it accepts only already initialized instances of the FederatedNode
        object.
        
        Parameters
        ----------
            nodess_list: list[FederatedNode] 
                The list containing all the initialized FederatedNode instances.
            model_list: list[nn.Module] 
                The list containing all the initialized nn.Module objects. 
                Note that conversion from nn.Module into the FederatedModel will occur 
                at the local node level.
            data_list (list[..., ....]): 
                The list containing train set and test set 
                wrapped in a hugging facr arrow_dataset.Dataset containers.
        
        Raises
        ------
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
        SOURCE: Communication-Efficient Learning of
        Deep Networks from Decentralized Data, H.B. McMahan et al.

        Parameters
        ----------
        nodes_data: list[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset] 
            A list containing train set and test set
            wrapped in a hugging face arrow_dataset.Dataset containers.
        -------------
        Returns:
        Int
            Returns 0 on the successful completion of the training."""
        
        # Initializing all the attributes using an instance of the Settings object.
        iterations = self.settings.iterations
        nodes_number = self.settings.number_of_nodes
        local_warm_start = self.settings.local_warm_start # Note: not implemeneted yet.
        nodes = [node for node in range(nodes_number)]
        sample_size = self.settings.sample_size
        
        # Initializing an instance of the Archiver class if enabled in the settings.
        if self.settings.enable_archiver == True:
            archive_manager = Archive_Manager(
                archive_manager = self.settings.archiver_settings,
                logger = orchestrator_logger)

        # Creating (empty) federated nodes.
        nodes_green = create_nodes(nodes, 
                                   self.settings.nodes_settings)


        # Creating a list of models for the nodes.
        model_list = self.model_initialization(nodes_number=nodes_number,
                                               model=self.central_net) # return deep copies of nets.
        
        # Initializing nodes -> loading the data and models onto empty nodes.
        nodes_green = self.nodes_initialization(nodes_list=nodes_green,
                                                model_list=model_list,
                                                data_list=nodes_data) # no deep copies of nets created at this stage
    
    # TRAINING PHASE ----- FEDAVG
        # create the pool of workers
        for iteration in range(iterations):
            orchestrator_logger.info(f"Iteration {iteration}")
            weights = {}
            # Sampling nodes and asynchronously apply the function
            sampled_nodes = sample_nodes(nodes_green, 
                                            sample_size=sample_size, 
                                            orchestrator_logger=orchestrator_logger) # SAMPLING FUNCTION -> CHANGE IF NEEDED
            if self.batch_job:
                for batch in Helpers.chunker(sampled_nodes, size=self.batch):
                    with Pool(sample_size) as pool:
                        results = [pool.apply_async(train_nodes, (node,)) for node in batch]
                        # consume the results
                        for result in results:
                            node_id, model_weights = result.get()
                            weights[node_id] = model_weights
            else:
                with Pool(sample_size) as pool:
                    results = [pool.apply_async(train_nodes, (node,)) for node in sampled_nodes]
                    # consume the results
                    for result in results:
                        node_id, model_weights = result.get()
                        weights[node_id] = model_weights
            # Computing the average
            avg = Aggregators.compute_average(weights) # AGGREGATING FUNCTION -> CHANGE IF NEEDED
            # Updating the nodes
            for node in nodes_green:
                node.model.update_weights(avg)
            # Upadting the orchestrator
            self.central_model.update_weights(avg)

            # Passing results to the archiver -> only if so enabled in the settings.
            if self.settings.enable_archiver == True:
                archive_manager.archive_training_results(iteration = iteration,
                                                        central_model=self.central_model,
                                                        nodes=nodes_green)
            if self.full_debug == True:
                log_gpu_memory(iteration=iteration)

        orchestrator_logger.critical("Training complete")
        return 0
                        
