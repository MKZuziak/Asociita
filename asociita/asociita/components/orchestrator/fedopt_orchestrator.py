from asociita.components.orchestrator.generic_orchestrator import Orchestrator # Parent class import
import datasets, copy
from typing import Any, Union
from asociita.components.nodes.federated_node import FederatedNode
from asociita.models.pytorch.federated_model import FederatedModel
from asociita.utils.computations import Aggregators
from asociita.utils.handlers import Handler
from asociita.utils.loggers import Loggers
from asociita.utils.orchestrations import prepare_nodes, create_nodes, check_health, sample_nodes, train_nodes
from asociita.utils.optimizers import Optimizers
from asociita.components.archiver.archive_manager import Archive_Manager
from multiprocessing import Pool, Manager
from torch import nn

orchestrator_logger = Loggers.orchestrator_logger()


class Fedopt_Orchestrator(Orchestrator):
    def __init__(self, settings: dict) -> None:
        """Orchestrator is a central object necessary for performing the simulation.
        It connects the nodes, maintain the knowledge about their state and manages the
        multithread pool. generic_orchestrator.orchestrator is a parent to all more
        specific orchestrators.
        
        -------------
        Args
            settings (dict): dictionary object cotaining all the settings of the orchestrator.
       -------------
         Returns
            None"""
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

        -------------
        Args:
            nodes_data(list[..., ....]): list containing train set and test set
                wrapped in a hugging facr arrow_dataset.Dataset containers.
        -------------
        Returns:
            None"""
        
        # SETTINGS -> CHANGE IF NEEDED
        # GENERAL SETTINGS
        iterations = self.settings['iterations']
        nodes_number = self.settings['number_of_nodes']
        local_warm_start = self.settings["local_warm_start"]
        nodes = self.settings["nodes"]
        sample_size = self.settings["sample_size"]
        archive_manager = Archive_Manager(archive_manager = self.settings['archiver'],
                                          logger = orchestrator_logger)

        # OPTIMIZER SETTINGS
        optimizer_settings = self.settings["optimizer"]
        optimizer_name = optimizer_settings["name"]
        
        # CREATING FEDERATED NODES
        nodes_green = create_nodes(nodes, self.node_settings) # Exterior method / function, can override in childr.


        # CREATING LOCAL MODELS (that will be loaded onto nodes)
        model_list = self.model_initialization(nodes_number=nodes_number,
                                               model=self.central_net) # Exterior method / function, can overide in childr.
        
        # INITIALIZING ALL THE NODES
        nodes_green = self.nodes_initialization(nodes_list=nodes_green, # Exterion method / function, can overide in child.
                                                model_list=model_list,
                                                data_list=nodes_data,
                                                nodes_number=nodes_number)
        
        Optim = Optimizers(weights = self.central_model.get_weights()) # Setting up the Optim.
        
        # TRAINING PHASE ----- FEDOPT
        with Manager() as manager:
            # create the shared queue
            queue = manager.Queue()

            # create the pool of workers
            with Pool(sample_size) as pool:
                for iteration in range(iterations):
                    orchestrator_logger.info(f"Iteration {iteration}")
                    gradients = {}
                    
                    # Sampling nodes and asynchronously apply the function
                    sampled_nodes = sample_nodes(nodes_green, 
                                                 sample_size=sample_size,
                                                 orchestrator_logger=orchestrator_logger) # SAMPLING FUNCTION -> CHANGE IF NEEDED
                    results = [pool.apply_async(train_nodes, (node, 'gradients')) for node in sampled_nodes]
                    # consume the results
                    for result in results:
                        node_id, model_gradients = result.get()
                        gradients[node_id] = copy.deepcopy(model_gradients)
                    
                    # Computing the average of gradients
                    grad_avg = Aggregators.compute_average(gradients) # AGGREGATING FUNCTION -> CHANGE IF NEEDED
                    updated_weights = Optim.fed_optimize(optimizer=optimizer_name,
                                                         settings=optimizer_settings,
                                                         weights=self.central_model.get_weights(),
                                                         delta=grad_avg)
                    # Updating the nodes
                    for node in nodes_green:
                        node.model.update_weights(updated_weights)
                    # Upadting the orchestrator
                    self.central_model.update_weights(updated_weights)

                    archive_manager.archive_training_results(iteration = iteration,
                                                             central_model=self.central_model,
                                                             nodes=nodes_green)
        
        orchestrator_logger.critical("Training complete")