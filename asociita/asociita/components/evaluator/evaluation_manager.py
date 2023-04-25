from asociita.components.evaluator.or_evaluator import OR_Evaluator
from asociita.models.pytorch.federated_model import FederatedModel
from copy import deepcopy
import numpy as np
import math



class Evaluation_Manager():
    def __init__(self,
                settings: dict,
                model: FederatedModel) -> None:
        """Manages the process of evaluation. Creates an instance of 
        Evaluation_Manager object, that controls all the instances
        that perform evaluation."""

        self.settings = settings
        if settings['evaluation'].get("Shapley_OR"):
            self.flag_shap_or = True
        else:
            self.flag_shap_or = False
        
        if settings['evaluation'].get("LOO_OR"):
            self.flag_loo_or = True
        else:
            self.flag_loo_or = False

        if self.flag_shap_or or self.flag_loo_or:
            self.or_evaluator = OR_Evaluator(settings=settings,
                                             model=model)
    
    def track_gradients(self,
                        gradients):
        if self.flag_shap_or:
            self.or_evaluator.track_shapley(gradients=gradients)
        elif self.flag_loo_or: # This is called ONLY when we don't calculate Shapley, but we calculate LOO
            self.or_evaluator.track_loo(gradients=gradients)
    

    def calculate_results(self):
        results = {}
        if self.flag_shap_or:
            self.or_evaluator.calculate_shaply()
            results["Shapley_OneRound"] = self.or_evaluator.shapley_values
        
        if self.flag_loo_or:
            self.or_evaluator.calculate_loo()
            results["LOO_OneRound"] = self.or_evaluator.loo_values
        
        return results
