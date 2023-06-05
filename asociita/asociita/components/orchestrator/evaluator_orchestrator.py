# PARENT CLASS IMPORT
from asociita.components.orchestrator.generic_orchestrator import Orchestrator # Parent class import
# LIBRARY MODULES IMPORT
from asociita.utils.orchestrations import create_nodes, sample_nodes, train_nodes
from asociita.utils.computations import Aggregators
from asociita.utils.handlers import Handler
from asociita.utils.loggers import Loggers
from asociita.utils.orchestrations import create_nodes, sample_nodes, train_nodes
from asociita.utils.optimizers import Optimizers
from asociita.components.evaluator.evaluation_manager import Evaluation_Manager
from asociita.components.archiver.archive_manager import Archive_Manager
# ADDITIONAL IMPORTS
import datasets, copy
from multiprocessing import Pool, Manager


orchestrator_logger = Loggers.orchestrator_logger()


class Evaluator_Orchestrator(Orchestrator):
    def __init__(self, settings: dict) -> None:
        """Orchestrator is a central object necessary for performing the simulation.
        It connects the nodes, maintain the knowledge about their state and manages the
        multithread pool. generic_orchestrator.orchestrator is a parent to all more
        specific orchestrators. Evaluator_Orchestrator additionally performs a 
        evaluation of the clients contribution, according to the passed settings.
        
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
        
        # 1. SETTINGS -> CHANGE IF NEEDED
        # GENERAL SETTINGS
        iterations = self.settings['iterations'] # Number of iterations of the Fed training, int.
        nodes_number = self.settings['number_of_nodes'] # Number of nodes prepared for sampling, int.
        local_warm_start = self.settings["local_warm_start"] # Local warm start for pre-trained models - not implemented yet.
        nodes = self.settings["nodes"] # List of nodes, list[int]
        sample_size = self.settings["sample_size"] # Size of the sample, int.
        # OPTIMIZER SETTINGS
        optimizer_settings = self.settings["optimizer"] # Dict containing instructions for the optimizer, dict.
        optimizer_name = optimizer_settings["name"] # Name of the optimizer, e.g. FedAdagard, dict.
        

        # 2. SET-UP PHASE -> CHANGE IF NEEDED
        # SETTING-UP EVALUATION MANAGER
        evaluation_manager = Evaluation_Manager(settings = self.settings,
                                                model = self.central_model,
                                                nodes = nodes,
                                                iterations = iterations)
        
        archive_manager = Archive_Manager(archive_manager = self.settings['archiver'],
                                          logger = orchestrator_logger)
        self.metrics_save_path = self.settings['metrics_save_path']

        # CREATING FEDERATED NODES
        nodes_green = create_nodes(nodes, self.node_settings)
        # CREATING LOCAL MODELS (that will be loaded onto nodes)
        model_list = self.model_initialization(nodes_number=nodes_number,
                                               model=self.central_net)
        # INITIALIZING ALL THE NODES
        nodes_green = self.nodes_initialization(nodes_list=nodes_green,
                                                model_list=model_list,
                                                data_list=nodes_data,
                                                nodes_number=nodes_number)
        # SETTING UP THE OPTIMIZER
        Optim = Optimizers(weights = self.central_model.get_weights(),
                           settings=optimizer_settings)

        # 3. TRAINING PHASE ----- FEDOPT
        with Manager() as manager:
            # create the shared queue
            queue = manager.Queue()
            # create the pool of workers
            with Pool(sample_size) as pool:
                for iteration in range(iterations):
                    orchestrator_logger.info(f"Iteration {iteration}")
                    gradients = {}
                    evaluation_manager.preserve_previous_model(previous_model = self.central_model) # Preserving last central model
                    evaluation_manager.preserve_previous_optimizer(previous_optimizer = Optim) # Preserving last optimizer settings (together with momentum)
                    
                    # Sampling nodes and asynchronously apply the function
                    sampled_nodes, sampled_idx = sample_nodes(nodes_green, 
                                                 sample_size=sample_size,
                                                 orchestrator_logger=orchestrator_logger,
                                                 return_aux=True) # SAMPLING FUNCTION -> CHANGE IF NEEDED
                    results = [pool.apply_async(train_nodes, (node, 'gradients')) for node in sampled_nodes]
                    # consume the results
                    for result in results:
                        node_id, model_gradients = result.get()
                        gradients[node_id] = copy.deepcopy(model_gradients)
                    
                    # Computing the average of gradients
                    grad_avg = Aggregators.compute_average(gradients) # AGGREGATING FUNCTION -> CHANGE IF NEEDED
                    updated_weights = Optim.fed_optimize(weights=self.central_model.get_weights(),
                                                         delta=grad_avg)
                    self.central_model.update_weights(updated_weights) # Updating the central model
                    evaluation_manager.preserve_updated_model(updated_model = self.central_model)

                    # TRACKING GRADIENTS FOR EVALUATION
                    evaluation_manager.track_results(gradients = gradients,
                                                     nodes_in_sample = sampled_nodes,
                                                     iteration = iteration,
                                                     optimizer = Optim) # TRACKING FUNCTION -> CHANGE IF NEEDED
                    
                    ### WEIGHTS UPDATE
                    # Updating the nodes
                    for node in nodes_green:
                        node.model.update_weights(updated_weights)
                    # Upadting the orchestrator                 
                    archive_manager.archive_training_results(iteration = iteration,
                                                             central_model=self.central_model,
                                                             nodes=nodes_green)
        # 4. FINALIZING PHASE
        # EVALUATING THE RESULTS
        
        
        psi = evaluation_manager.finalize_tracking()
        print(psi)
        
        orchestrator_logger.critical("Training complete")