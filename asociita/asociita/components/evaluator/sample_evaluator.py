from asociita.utils.computations import Aggregators
import numpy as np
import copy

class Sample_Evaluator():
    def __init__(self,
                 nodes: list,
                 iterations: int) -> None:
        self.psi = {node: np.float64(0) for node in nodes} # Hash map containing all the nodes and their respective marginal contribution values.
        self.partial_psi = {round:{} for round in range(iterations)} # Hash map containing all the partial psi for each sampled subset.


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
            # marginal_sample = copy.deepcopy(nodes_in_sample)
            # del marginal_sample[node_id]
            
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
        for iteration, iteration_results in self.partial_psi.items():
            for node, value in iteration_results.items():
                self.psi[node] += np.float64(value)
        return self.psi

