from asociita.utils.computations import Aggregators
from asociita.utils.computations import Subsets
import numpy as np
import copy
import math

def select_gradients(gradients,
                     query: list,
                     in_place: bool = False):
    """Select gradients given the list of quered nodes"""
    q_gradients = {}
    if in_place == False:
        gradients_copy = copy.deepcopy(gradients) #TODO: Inspect whether making copt is realy necc.
    for id, grad in gradients_copy.items():
        if id in query:
            q_gradients[id] = grad
    return q_gradients


class Sample_Evaluator():
    def __init__(self,
                 nodes: list,
                 iterations: int) -> None:
        self.psi = {node: np.float64(0) for node in nodes} # Hash map containing all the nodes and their respective marginal contribution values.
        self.partial_psi = {round:{node: np.float64(0) for node in nodes} for round in range(iterations)} # Hash map containing all the partial psi for each sampled subset.


    def update_psi(self,
                   gradients,
                   nodes_in_sample,
                   optimizer,
                   iteration: int,
                   final_model,
                   previous_model):
        final_model_score = final_model.evaluate_model()[1]
        
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


    def calculate_final_psi(self):
        for iteration_results in self.partial_psi.values():
            for node, value in iteration_results.items():
                self.psi[node] += np.float64(value)
        return (self.partial_psi, self.psi)


class Sample_Shapley_Evaluator():
    def __init__(self,
                 nodes: list,
                 iterations: int) -> None:
        self.shapley = {node: np.float64(0) for node in nodes} # Hash map containing all the nodes and their respective marginal contribution values.
        self.partial_shapley = {round:{node: np.float64(0) for node in nodes} for round in range(iterations)} # Hash map containing all the partial psi for each sampled subset.
    

    def update_shap(self,
                    gradients,
                    nodes_in_sample,
                    optimizer,
                    iteration: int,
                    previous_model):
            
            operation_counter = 1
            number_of_operations = 2 ** (len(nodes_in_sample)) - 1
            recorded_values = {} # Maps every coalition to it's value, implemented to decrease the complexity.
            nodes_in_sample = [node.node_id for node in nodes_in_sample] # Converting list of FederatedNode objects to the int representing their identiity.
            superset = Subsets.form_superset(nodes_in_sample, return_dict=True)

            for node in nodes_in_sample:
                shap = 0.0
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
                        s_wo_i_score = s_wo_i_model.evaluate_model()[1]
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
                        s_w_i_score = s_w_i_model.evaluate_model()[1]
                        recorded_values[s_w_i] = s_w_i_score
                        operation_counter += 1
                    
                    sample_pos = math.comb((len(nodes_in_sample) - 1), len(s_wo_i)) # Find the total number of possibilities to choose k things from n items:

                    divisor = 1 / sample_pos
                    shap += divisor * (s_w_i_score - s_wo_i_score)
                self.partial_shapley[iteration][node] =  shap / (len(nodes_in_sample))
    

    def calculate_final_shap(self):
        for iteration_results in self.partial_shapley.values():
            for node, value in iteration_results.items():
                self.shapley[node] += np.float64(value)
        return (self.partial_shapley, self.shapley)