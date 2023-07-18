from asociita.utils.computations import Aggregators
from asociita.utils.computations import Subsets
import numpy as np
import copy
import math
from multiprocessing import Pool, Manager
from asociita.models.pytorch.federated_model import FederatedModel
from collections import OrderedDict
from _collections_abc import Generator
from asociita.utils.optimizers import Optimizers


def chunker(seq: iter, size: int) -> Generator:
    """Helper function for splitting an iterable into a number
    of chunks each of size n. If the iterable can not be splitted
    into an equal number of chunks of size n, chunker will return the
    left-overs as the last chunk.
        
    Parameters
    ----------
    sqe: iter
        An iterable object that needs to be splitted into n chunks.
    size: int
        A size of individual chunk
    Returns
    -------
    Generator
        A generator object that can be iterated to obtain chunks of the original iterable.
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def select_gradients(gradients: OrderedDict,
                     query: list,
                     in_place: bool = False):
    """Helper function for selecting gradients that of nodes that are
    in the query.

    Parameters
    ----------
    sqe: OrderedDict
        An OrderedDict containing all the gradients.
    query: list
        A size containing ids of the searched nodes.
    in_place: bool, default to False
        No description
    Returns
    -------
    Generator
        A generator object that can be iterated to obtain chunks of the original iterable.
    """
    q_gradients = {}
    # if in_place == False:
    #     gradients_copy = copy.deepcopy(gradients) #TODO: Inspect whether making copy is realy necc.
    for id, grad in gradients.items():
        if id in query:
            q_gradients[id] = grad
    return q_gradients


class Sample_Evaluator():
    """Sample evaluator is used to establish the marginal contribution of each sampled
    client to the general value of the global model. Basic Sample Evaluator is able to
    assess the Leave-one-out value for every client included in the sample. It is also
    able to sum the marginal values to obain a final Leave-one-out value."""
    
    def __init__(self,
                 nodes: list,
                 iterations: int) -> None:
        """Constructor for the Sample Evaluator Class. Initializes empty
        hash tables for LOO value for each iteration as well as hash table
        for final LOO values.
        
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
        
        self.psi = {node: np.float64(0) for node in nodes} # Hash map containing all the nodes and their respective marginal contribution values.
        self.partial_psi = {round:{node: np.float64(0) for node in nodes} for round in range(iterations)} # Hash map containing all the partial psi for each sampled subset.


    def update_psi(self,
                   gradients: OrderedDict,
                   nodes_in_sample: list,
                   optimizer: Optimizers,
                   iteration: int,
                   final_model: FederatedModel,
                   previous_model: FederatedModel,
                   return_coalitions: bool = True):
        """Method used to track_results after each training round.
        Given the graidnets, ids of the nodes included in sample,
        last version of the optimizer, previous version of the model
        and the updated version of the model, it calculates values of
        all the marginal contributions using Leave-one-out method.
        
        Parameters
        ----------
        gradients: OrderedDict
            An OrderedDict containing gradients of the sampled nodes.
        nodes_in_sample: list
            A list containing id's of the nodes that were sampled.
        previous_optimizer: Optimizers
            An instance of the asociita.Optimizers class.
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

        # Evaluating the performance of the final model.
        final_model_score = final_model.evaluate_model()[1]
        
        recorded_values[tuple(gradients.keys())] = final_model_score
        
        for node in nodes_in_sample:
            node_id = node.node_id
            
            # Deleting gradients of node i from the sample.
            marginal_gradients = copy.deepcopy(gradients)
            del marginal_gradients[node_id] 
            # Cloning the last optimizer
            marginal_optim = copy.deepcopy(optimizer)

            # Reconstrcuting the marginal model
            marginal_model = copy.deepcopy(previous_model)
            marginal_grad_avg = Aggregators.compute_average(marginal_gradients) # AGGREGATING FUNCTION -> CHANGE IF NEEDED
            marginal_weights = marginal_optim.fed_optimize(weights=marginal_model.get_weights(),
                                                           delta=marginal_grad_avg)
            marginal_model.update_weights(marginal_weights)
            marginal_model_score = marginal_model.evaluate_model()[1]
            self.partial_psi[iteration][node_id] = final_model_score - marginal_model_score
            
            recorded_values[tuple(marginal_gradients.keys())] = marginal_model_score
            
        if return_coalitions == True:
            return recorded_values


    def calculate_final_psi(self) -> tuple[dict[int: dict], dict[int: float]]:
        """Method used to sum up all the partial LOO scores to obtain
        a final LOO score for each client.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        tuple[dict[int: dict], dict[int: float]]
        """
        
        for iteration_results in self.partial_psi.values():
            for node, value in iteration_results.items():
                self.psi[node] += np.float64(value)
        return (self.partial_psi, self.psi)


class Sample_Shapley_Evaluator():
    """Sample evaluator is used to establish the marginal contribution of each sampled
    client to the general value of the global model using Shapley Value as a method of
    assesment. Shapley Sample Evaluator is able to assess the Shapley value for every 
    client included in the sample. It is also able to sum the marginal values to obain 
    a final Shapley values."""
    
    def __init__(self,
                 nodes: list,
                 iterations: int) -> None:
        """Constructor for the Shapley Sample Evaluator Class. Initializes empty
        hash tables for Shapley value for each iteration as well as hash table
        for final Shapley values.
        
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
        self.shapley = {node: np.float64(0) for node in nodes} # Hash map containing all the nodes and their respective marginal contribution values.
        self.partial_shapley = {round:{node: np.float64(0) for node in nodes} for round in range(iterations)} # Hash map containing all the partial psi for each sampled subset.
    

    def update_shap(self,
                    gradients: OrderedDict,
                    nodes_in_sample: list,
                    optimizer: Optimizers,
                    iteration: int,
                    previous_model: FederatedModel,
                    return_coalitions: bool = True):
        """Method used to track_results after each training round.
        Given the graidnets, ids of the nodes included in sample,
        last version of the optimizer, previous version of the model
        and the updated version of the model, it calculates values of
        all the marginal contributions using Shapley value.
        
        Parameters
        ----------
        gradients: OrderedDict
            An OrderedDict containing gradients of the sampled nodes.
        nodes_in_sample: list
            A list containing id's of the nodes that were sampled.
        previous_optimizer: Optimizers
            An instance of the asociita.Optimizers class.
        iteration: int
            The current iteration.
        previous_model: FederatedModel
            An instance of the FederatedModel object.
        updated_model: FederatedModel
            An instance of the FederatedModel object.
        return_coalitions: bool, default to True
            If set to True, method will return value-mapping for every coalition.
        Returns
        -------
        None
        """
        
        # Operations counter to track the progress of the calculations.
        operation_counter = 1
        number_of_operations = 2 ** (len(nodes_in_sample)) - 1

        # Maps every coalition to it's value, implemented to decrease the time complexity.
        recorded_values = {}
        
        # Converting list of FederatedNode objects to the int representing their identiity.
        nodes_in_sample = [node.node_id for node in nodes_in_sample] 
        # Forming superset of all the possible coalitions.
        superset = Subsets.form_superset(nodes_in_sample, return_dict=True)


        for node in nodes_in_sample:
            shap = 0.0
            # Select subsets that do not contain agent i
            S = Subsets.select_subsets(coalitions = superset, searched_node = node)
            for s in S.keys():
                s_wo_i = tuple(sorted(s)) # Subset s without the agent i
                s_w_i = tuple(sorted(s + (node, ))) # Subset s with the agent i

                # Evaluating the performance of the model trained without client i
                if recorded_values.get(s_wo_i):
                    s_wo_i_score = recorded_values[s_wo_i]
                else:
                    print(f"{operation_counter} of {number_of_operations}: forming and evaluating subset {s_wo_i}") #TODO: Consider if a logger would not be better option.
                    s_wo_i_gradients = select_gradients(gradients = gradients, query = s_wo_i)
                    s_wo_i_optim = copy.deepcopy(optimizer)
                    s_wo_i_model = copy.deepcopy(previous_model)
                    s_wo_i_grad_avg = Aggregators.compute_average(s_wo_i_gradients)
                    s_wo_i_weights = s_wo_i_optim.fed_optimize(weights = s_wo_i_model.get_weights(), delta = s_wo_i_grad_avg)
                    s_wo_i_model.update_weights(s_wo_i_weights)
                    s_wo_i_score = s_wo_i_model.quick_evaluate()[1]
                    recorded_values[s_wo_i] = s_wo_i_score
                    operation_counter += 1

                # Evaluating the performance of the model trained with client i
                if recorded_values.get(s_w_i):
                    s_w_i_score = recorded_values[s_w_i]
                else:
                    print(f"{operation_counter} of {number_of_operations}: forming and evaluating subset {s_w_i}") #TODO: Consider if a logger would not be better option.
                    s_w_i_gradients = select_gradients(gradients = gradients, query = s_w_i)
                    s_w_i_optim = copy.deepcopy(optimizer)
                    s_w_i_model = copy.deepcopy(previous_model)
                    s_w_i_grad_avg = Aggregators.compute_average(s_w_i_gradients)
                    s_w_i_weights = s_w_i_optim.fed_optimize(weights = s_w_i_model.get_weights(), delta = s_w_i_grad_avg)
                    s_w_i_model.update_weights(s_w_i_weights)
                    s_w_i_score = s_w_i_model.quick_evaluate()[1]
                    recorded_values[s_w_i] = s_w_i_score
                    operation_counter += 1
                
                sample_pos = math.comb((len(nodes_in_sample) - 1), len(s_wo_i)) # Find the total number of possibilities to choose k things from n items:

                divisor = 1 / sample_pos
                shap += divisor * (s_w_i_score - s_wo_i_score)
            self.partial_shapley[iteration][node] =  shap / (len(nodes_in_sample))

            if return_coalitions == True:
                return recorded_values
        

    def update_shap_multip(self,
                           gradients: OrderedDict,
                           nodes_in_sample: list,
                           optimizer: Optimizers,
                           iteration: int,
                           previous_model: FederatedModel,
                           number_of_workers: int = 30,
                           return_coalitions: bool = True):
        """Method used to track_results after each training round.
        Update_shap_multip is a default method used to calculate
        Shapley round, as it uses a number of workers to complete a task.
    
        Given the graidnets, ids of the nodes included in sample,
        last version of the optimizer, previous version of the model
        and the updated version of the model, it calculates values of
        all the marginal contributions using Shapley value.
        
        Parameters
        ----------
        gradients: OrderedDict
            An OrderedDict containing gradients of the sampled nodes.
        nodes_in_sample: list
            A list containing id's of the nodes that were sampled.
        previous_optimizer: Optimizers
            An instance of the asociita.Optimizers class.
        iteration: int
            The current iteration.
        previous_model: FederatedModel
            An instance of the FederatedModel object.
        updated_model: FederatedModel
            An instance of the FederatedModel object.
        number_of_workers: int, default to 50
            A number of workers that will simultaneously work on a task.
        return_coalitions: bool, default to True
            If set to True, method will return value-mapping for every coalition.
        Returns
        -------
        None
        """
        coalition_results = {}
        nodes_in_sample = [node.node_id for node in nodes_in_sample] 
        superset = Subsets.form_superset(nodes_in_sample, return_dict=False)
        # Operations counter to track the progress of the calculations.
        operation_counter = 0
        number_of_operations = 2 ** (len(nodes_in_sample)) - 1
        if len(superset) < number_of_workers:
            number_of_workers = len(superset)
        chunked = chunker(seq = superset, size = number_of_workers)
        with Pool(number_of_workers) as pool:
            for chunk in chunked:
                results = [pool.apply_async(self.establish_value, (coalition, 
                                                                    copy.deepcopy(gradients),
                                                                    copy.deepcopy(optimizer),
                                                                    copy.deepcopy(previous_model))) for coalition in chunk]
                for result in results:
                    coalition, score = result.get()
                    coalition_results[tuple(sorted(coalition))] = score
                operation_counter += len(chunk)
                print(f"Completed {operation_counter} out of {number_of_operations} operations")
        print("Finished evaluating all of the coalitions. Commencing calculation of individual Shapley values.")
        for node in nodes_in_sample:
            shap = 0.0
            S = Subsets.select_subsets(coalitions = superset, searched_node = node)
            for s in S:
                s_wo_i = tuple(sorted(s)) # Subset s without the agent i
                s_copy = copy.deepcopy(s)
                s_copy.append(node)
                s_w_i = tuple(sorted(s_copy)) # Subset s with the agent i
                sample_pos = math.comb((len(nodes_in_sample) - 1), len(s_wo_i))
                divisor = float(1 / sample_pos)
                shap += float(divisor * (coalition_results[s_w_i] - coalition_results[s_wo_i]))
            
            self.partial_shapley[iteration][node] =  float(shap / (len(nodes_in_sample)))
        
        if return_coalitions == True:
            return coalition_results


    def establish_value(self,
                        coalition: list,
                        gradients: OrderedDict,
                        optimizer: Optimizers,
                        model: FederatedModel) -> tuple[list, float]:
        """Helper method used to establish a value of a particular coalition.
        Called asynchronously in multiprocessed version of the self.update_shap_multip()
        method.
        
        Parameters
        ----------
        None
        gradients: OrderedDict
            An OrderedDict containing gradients of the sampled nodes.
        coalition: list
            A list containing id's of the nodes that were sampled.
        optimizer: Optimizers
            An instance of the asociita.Optimizers class.
        model: FederatedModel
            An instance of the FederatedModel object.
        Returns
        -------
        tuple[list, float]
        """
        gradients = select_gradients(gradients = gradients,
                                     query = coalition)
        grad_avg = Aggregators.compute_average(gradients)
        weights = optimizer.fed_optimize(weights = model.get_weights(), delta = grad_avg)
        model.update_weights(weights)
        score = model.quick_evaluate()[1]
        return (coalition, score)


    def calculate_final_shap(self):
        """Method used to sum up all the partial Shapley values to obtain
        a final Shapley score for each client.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        tuple[dict[int: dict], dict[int: float]]
        """
        for iteration_results in self.partial_shapley.values():
            for node, value in iteration_results.items():
                self.shapley[node] += np.float64(value)
        return (self.partial_shapley, self.shapley)