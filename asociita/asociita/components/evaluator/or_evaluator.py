# LIBRARY MODULES IMPORT
from asociita.utils.computations import Subsets
from asociita.models.pytorch.federated_model import FederatedModel
from asociita.utils.computations import Aggregators
from asociita.utils.optimizers import Optimizers
from copy import deepcopy
# ADDITIONAL IMPORTS
import math


class OR_Evaluator():
    def __init__(self,
                 settings: dict,
                 model: FederatedModel) -> None:
        """A one_round evaluator that firstly collects all the models reconstructed
        from gradients, and then perform an evaluation according to the chosen metric.
        -------------
        Args
            settings (dict): dictionary object cotaining all the settings of the orchestrator.
            model (FederatedModel): a primary (initial) model that will be deep-copied for each coalition.
       -------------
         Returns
            None"""
        self.settings = settings
        self.evaluation = settings['evaluation']
        self.recorded_values = {} # Maps every coalition to it's value, implemented to decrease the complexity.
        self.shapley_or_recon = None
        
        # Creates coalitions for Shapley, if so indicated in the settings.
        if self.evaluation.get("Shapley_OR"):
            self.shapley_values = {node:float(0) for node in settings['nodes']} # Final list (of Shapley evaluation)
            self.shapley_or_recon = Subsets.form_superset(settings['nodes'], return_dict=False)
            self.shapley_or_recon = {tuple(coalition) : deepcopy(model) for 
                                     coalition in self.shapley_or_recon}
        
        # Creates coalition for Leave-one-out, if so indicated in the settings.
        if self.evaluation.get("LOO_OR"):
            self.loo_values = {node:float(0) for node in settings['nodes']} # Final list (of LOO evaluation)
            if self.shapley_or_recon: # If we already have coalition for shapleys values, we can use model of those
                self.loo_or_recon = {coalition: model for coalition, model in self.shapley_or_recon.items()
                                     if len(coalition) >= (settings["number_of_nodes"] - 1)}
            else:
                self.loo_or_recon = Subsets.form_loo_set(settings['nodes'], return_dict=False) # Else we have to create a new one.
                self.loo_or_recon = {tuple(coaliton) : deepcopy(model) for
                                     coaliton in self.loo_or_recon}


    def track_shapley(self,
                      gradients: dict):
        """A method that allows to collect gradients from the t-th round of the training and
        update all the models in every coalition of interest.
        
        -------------
        Args
            gradients (dict): Dictionary mapping each node to its respective gradients.
       -------------
         Returns
            None"""
        
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
    

    def track_loo(self,
                   gradients: dict):
        """A method that allows to collect gradients from the t-th round of the training and
        update all the models in every coalition of interest. Note that it should be called
        ONLY when we DO NOT track Shapley values. Otherwise, models used for LOO evaluation
        will be a shallow copy of some of the models used for Shapley valuation and SHOULD NOT
        be updated again.
        
        -------------
        Args
            gradients (dict): Dictionary mapping each node to its respective gradients.
       -------------
         Returns
            None"""
        
        # Establishing gradients for all possible coalitions in N
        delta_s = {coalition: float(0) for coalition in self.loo_or_recon.keys()}
        for coalition in delta_s:
            specific_gradients = {}
            for member in coalition:
                specific_gradients[member] = gradients[member]
            delta_s[coalition] = Aggregators.compute_average(specific_gradients)
        
        # Upadting models of all possible coalitions in N
        for coalition in self.loo_or_recon:
            model_s_t = self.loo_or_recon[coalition]
            delta_s_t = delta_s[coalition]
            updated_weights = Optimizers.SimpleFedopt(weights=model_s_t.get_weights(),
                                                      delta=delta_s_t,
                                                      learning_rate=0.99)
            self.loo_or_recon[coalition].update_weights(updated_weights)

    def calculate_shaply(self):
        """Calculates Shapley values.
        -------------
        Args
            None
       -------------
         Returns
            None"""
        N = self.settings['number_of_nodes']
        operation_counter = 1
        number_of_operations = 2 ** N - 1
        for node in self.shapley_values:
            shapley_value = float(0)
            subsets = Subsets.select_subsets(coalitions=self.shapley_or_recon,
                                            searched_node=node)
            for subset in subsets.keys():
                subset_without_i = tuple(sorted(subset))
                subset_with_i = tuple(sorted(subset + (node, )))

                if subset_without_i in self.recorded_values:
                    score_without_i = self.recorded_values[subset_without_i]
                else:
                    print(f"{operation_counter} of {number_of_operations}: forming and evaluating subset {subset_without_i}")
                    model_without_i = self.shapley_or_recon[subset_without_i]
                    score_without_i = model_without_i.evaluate_model()[1]
                    self.recorded_values[subset_without_i] = score_without_i
                    operation_counter += 1

                if subset_with_i in self.recorded_values:
                    score_with_i = self.recorded_values[subset_with_i]
                else:
                    print(f"{operation_counter} of {number_of_operations}: forming and evaluating subset {subset_with_i}")
                    model_with_i = self.shapley_or_recon[subset_with_i]
                    score_with_i = model_with_i.evaluate_model()[1]
                    self.recorded_values[subset_with_i] = score_with_i
                    operation_counter += 1
                
                summand = (score_with_i - score_without_i) /  math.comb((N-1), len(subset))
                shapley_value += summand
            
            self.shapley_values[node] = shapley_value
    

    def calculate_loo(self):
        """Calculates Leave-one-out values.
        -------------
        Args
            None
       -------------
         Returns
            None"""
        
        all_nodes = deepcopy(self.settings['nodes'])
        general_subset = tuple(self.settings['nodes'])
        if general_subset in self.recorded_values:
            general_score = self.recorded_values[general_subset]
        else:
            print(f"Forming and evaluating subset {general_subset}")
            general_model = self.loo_or_recon[general_subset]
            general_score = general_model.evaluate_model()[1]
            self.recorded_values[general_subset] = general_score
        
        for node in self.loo_values:
            subset_without_i = deepcopy(all_nodes)
            subset_without_i.remove(node)
            subset_without_i = tuple(sorted(subset_without_i))
            if subset_without_i in self.recorded_values:
                score_without_i = self.recorded_values[subset_without_i]
            else:
                print(f"Forming and evaluating subset {subset_without_i}")
                model_without_i = self.loo_or_recon[subset_without_i]
                score_without_i = model_without_i.evaluate_model()[1]
                self.recorded_values[subset_without_i] = score_without_i
            
            self.loo_values[node] = general_score - score_without_i
            
