import datasets
import copy
from asociita.components.orchestrator.generic_orchestrator import Orchestrator
from asociita.utils.computations import Aggregators
from asociita.utils.loggers import Loggers
from asociita.utils.orchestrations import create_nodes, sample_nodes, train_nodes
from asociita.utils.optimizers import Optimizers
from asociita.components.archiver.archive_manager import Archive_Manager
from multiprocessing import Pool, Manager
from asociita.components.settings.settings import Settings
from torch import nn

# set_start_method set to 'spawn' to ensure compatibility across platforms.
orchestrator_logger = Loggers.orchestrator_logger()
from multiprocessing import set_start_method
set_start_method("spawn", force=True)


class Fedopt_Orchestrator(Orchestrator):
    """Orchestrator is a central object necessary for performing the simulation.
        It connects the nodes, maintain the knowledge about their state and manages the
        multithread pool. Fedopt orchestrator is a child class of the Generic Orchestrator.
        Unlike its parent, FedOpt Orchestrator performs a training using Federated Optimization
        - pseudo-gradients from the models and momentum."""
    

    def __init__(self, 
                 settings: Settings) -> None:
        """Orchestrator is initialized by passing an instance
        of the Settings object. Settings object contains all the relevant configurational
        settings that an instance of the Orchestrator object may need to complete the simulation.
        FedOpt orchestrator additionaly requires a configuration passed to the Optimizer upon
        its initialization.
        
        Parameters
        ----------
        settings : Settings 
            An instance of the Settings object cotaining all the settings of the orchestrator.
            The FedOpt orchestrator additionaly requires the passed object to contain a 
            configuration for the Optimizer.
       
       Returns
       -------
       None
       """
        super().__init__(settings)
    

    def train_protocol(self,
                nodes_data: list[datasets.arrow_dataset.Dataset, 
                datasets.arrow_dataset.Dataset]) -> None:
        """"Performs a full federated training according to the initialized
        settings. The train_protocol of the fedopt.orchestrator.Fedopt_Orchestrator
        follows a popular FedAvg generalisation, FedOpt. Instead of weights from each
        clients, it aggregates gradients (understood as a difference between the weights
        of a model after all t epochs of the local training) and aggregates according to 
        provided rule.
        SOURCE: Adaptive Federated Optimization, S.J. Reddi et al.

        Parameters
        ----------
        nodes_data: list[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset]: 
            A list containing train set and test set wrapped 
            in a hugging face arrow_dataset.Dataset containers.
        
        Returns
        -------
        int
            Returns 0 on the successful completion of the training.
        """
        # Initializing all the attributes using an instance of the Settings object.
        iterations = self.settings.iterations
        nodes_number = self.settings.number_of_nodes
        local_warm_start = self.settings.local_warm_start # Note: not implemented yet.
        nodes = [node for node in range(nodes_number)]
        sample_size = self.settings.sample_size
        
        # Initializing an instance of the Archiver class if enabled in the settings.
        if self.settings.enable_archiver == True:
            archive_manager = Archive_Manager(
                archive_manager = self.settings.archiver_settings,
                logger = orchestrator_logger)
        
        # Initializing an instance of the Optimizer class object.
        optimizer_settings = self.settings.optimizer_settings
        Optim = Optimizers(weights = self.central_model.get_weights(),
                           settings=optimizer_settings)
        
        # Creating (empty) federated nodes.
        nodes_green = create_nodes(nodes, self.settings.nodes_settings)
        
        # Creating a list of models for the nodes.
        model_list = self.model_initialization(nodes_number=nodes_number,
                                               model=self.central_net)
        
        # Initializing nodes -> loading the data and models onto empty nodes.
        nodes_green = self.nodes_initialization(nodes_list=nodes_green,
                                                model_list=model_list,
                                                data_list=nodes_data)

        # 3. TRAINING PHASE ----- FEDOPT
        with Manager() as manager:
            queue = manager.Queue() # creates a shared queue
            # create the pool of workers
            with Pool(sample_size) as pool:
                for iteration in range(iterations):
                    orchestrator_logger.info(f"Iteration {iteration}")
                    gradients = {}
                    # Sampling nodes and asynchronously apply the function
                    sampled_nodes = sample_nodes(nodes_green, 
                                                 sample_size=sample_size,
                                                 orchestrator_logger=orchestrator_logger,
                                                 return_aux=False) # SAMPLING FUNCTION -> CHANGE IF NEEDED
                    results = [pool.apply_async(train_nodes, (node, 'gradients')) for node in sampled_nodes]
                    # consume the results
                    for result in results:
                        node_id, model_gradients = result.get()
                        gradients[node_id] = copy.deepcopy(model_gradients)
                    # Computing the average of gradients
                    grad_avg = Aggregators.compute_average(gradients) # AGGREGATING FUNCTION -> CHANGE IF NEEDED
                    # Upadting the weights using gradients and momentum
                    updated_weights = Optim.fed_optimize(weights=self.central_model.get_weights(),
                                                         delta=grad_avg)
                    # Updating the orchestrator
                    self.central_model.update_weights(updated_weights)
                    # Updating the nodes
                    for node in nodes_green:
                        node.model.update_weights(updated_weights)         
                   
                    # Passing results to the archiver -> only if so enabled in the settings.
                    if self.settings.enable_archiver == True:
                        archive_manager.archive_training_results(iteration = iteration,
                                                                central_model=self.central_model,
                                                                nodes=nodes_green)

        orchestrator_logger.critical("Training complete")
        return 0