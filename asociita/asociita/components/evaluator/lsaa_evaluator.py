import numpy as np
import copy
from asociita.models.pytorch.federated_model import FederatedModel
from asociita.utils.optimizers import Optimizers
from asociita.utils.computations import Aggregators
from collections import OrderedDict


class LSAA():
    """LSAA is used to establish the marginal contribution of each sampled
    client to the general value of the global model. LSAA is based on the assumption
    that we can detect the influence that a sampled client has on a general model
    by testing a scenario in which we have more-alike clients included in the sample."""
    
    def __init__(self,
                 nodes: list,
                 iterations: int) -> None:
        """Constructor for the LSAA. Initializes empty
        hash tables for LSAA value for each iteration as well as hash table
        for final LSAA values.
        
        Parameters
        ----------
        nodes: list
            A list containing ids of all the nodes engaged in the training.
        iterations: int
            A number of training iterations
        Returns
        -------
        None
        """
        
        self.lsaa = {node: np.float64(0) for node in nodes} # Hash map containing all the nodes and their respective marginal contribution values.
        self.partial_lsaa = {round:{node: np.float64(0) for node in nodes} for round in range(iterations)} # Hash map containing all the partial psi for each sampled subset.
    

    def update_lsaa(self,
                   gradients: OrderedDict,
                   nodes_in_sample: list,
                   optimizer: Optimizers,
                   search_length: int,
                   iteration: int,
                   final_model: FederatedModel,
                   previous_model: FederatedModel,
                   return_coalitions: bool = True):
        """Method used to track_results after each training round.
        Given the graidnets, ids of the nodes included in sample,
        last version of the optimizer, previous version of the model
        and the updated version of the model, it calculates values of
        all the marginal contributions using LSAA.
        
        Parameters
        ----------
        gradients: OrderedDict
            An OrderedDict containing gradients of the sampled nodes.
        nodes_in_sample: list
            A list containing id's of the nodes that were sampled.
        optimizer: Optimizers
            An instance of the asociita.Optimizers class.
        search length: int,
            A number of replicas that should be included in LSA search.
        iteration: int
            The current iteration.
        previous_model: FederatedModel
            An instance of the FederatedModel object.
        updated_model: FederatedModel
            An instance of the FederatedModel object.
        Returns
        -------
        None
        """
        
        recorded_values = {}

        for node in nodes_in_sample:
            lsaa_score = 0
            node_id = node.node_id
            # Deleting gradients of node i from the sample.
            marginal_gradients = copy.deepcopy(gradients)
            del marginal_gradients[node_id] 

            # Creating copies for the appended version
            appended_gradients = copy.deepcopy(marginal_gradients)
            appended_model = copy.deepcopy(previous_model)
            appended_optimizer = copy.deepcopy(optimizer)

            # Cloning the last optimizer
            marginal_optim = copy.deepcopy(optimizer)

            # Reconstrcuting the marginal model
            marginal_model = copy.deepcopy(previous_model)
            marginal_grad_avg = Aggregators.compute_average(marginal_gradients) # AGGREGATING FUNCTION -> CHANGE IF NEEDED
            marginal_weights = marginal_optim.fed_optimize(weights=marginal_model.get_weights(),
                                                        delta=marginal_grad_avg)
            marginal_model.update_weights(marginal_weights)
            marginal_model_score = marginal_model.quick_evaluate()[1]

            recorded_values[tuple(marginal_gradients.keys())] = marginal_model_score
            
            for phi in range(search_length):
                appended_gradients[(f"{phi + 1}_of_{node_id}")] = copy.deepcopy(gradients[node_id])
                # TODO: Change f"{phi + 1}_dummy_of_{node_id}" after debugging
            
            appended_grad_avg = Aggregators.compute_average(appended_gradients)
            appended_weights = appended_optimizer.fed_optimize(weights=appended_model.get_weights(),
                                                            delta = appended_grad_avg)
            appended_model.update_weights(appended_weights)
            appended_model_score = appended_model.quick_evaluate()[1]
            lsaa_score = appended_model_score - marginal_model_score
            recorded_values[tuple(appended_gradients.keys())] = appended_model_score
            
            self.partial_lsaa[iteration][node_id] = lsaa_score # Previously: lsaa_score / search_length
       
        if return_coalitions == True:
                return recorded_values
        
    
    def calculate_final_lsaa(self) -> tuple[dict[int: dict], dict[int: float]]:
        """Method used to sum up all the partial LOO scores to obtain
        a final LOO score for each client.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        tuple[dict[int: dict], dict[int: float]]
        """
        
        for iteration_results in self.partial_lsaa.values():
            for node, value in iteration_results.items():
                self.lsaa[node] += np.float64(value)
        return (self.partial_lsaa, self.lsaa)

