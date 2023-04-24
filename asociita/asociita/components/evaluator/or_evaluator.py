from asociita.utils.computations import Subsets
from asociita.models.pytorch.federated_model import FederatedModel
from asociita.utils.computations import Aggregators
from asociita.utils.optimizers import Optimizers
from copy import deepcopy


class OR_Evaluator():
    def __init__(self,
                 settings: dict,
                 model: FederatedModel) -> None:
        """A one_round evaluator that firstly collects all the models reconstructed
        from gradients, and then perform an evaluation according to the chosen metric."""
        self.settings = settings
        self.evaluation = settings['evaluation']
        if self.evaluation.get("Shapley_OR"):
            self.shapley_or_recon = Subsets.form_superset(settings['nodes'], return_dict=False)
            self.shapley_or_recon = {tuple(coalition) : deepcopy(model) for 
                                     coalition in self.shapley_or_recon}
    

    def track_shapley(self,
                      gradients):
        # Establishing gradients for all possible coalitions in N
        delta_s = Subsets.form_superset(self.settings["nodes"], return_dict=True)
        for coalition in delta_s:
            specific_gradients = {}
            for member in coalition:
                specific_gradients[member] = gradients[member]
            delta_s[coalition] = Aggregators.compute_average(specific_gradients)
        
        # Upadting models of all possible coalitions in N
        for coalition in self.shapley_or_recon:
            model_s_t = self.shapley_or_recon[coalition]
            delta_s_t = delta_s[coalition]
            updated_weights = Optimizers.SimpleFedopt(weights=model_s_t.get_weights(),
                                                                       delta=delta_s_t,
                                                                       learning_rate=0.99)
            self.shapley_or_recon[coalition].update_weights(updated_weights)