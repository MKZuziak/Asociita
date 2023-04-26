from asociita.components.evaluator.or_evaluator import OR_Evaluator
from asociita.models.pytorch.federated_model import FederatedModel


class Evaluation_Manager():
    def __init__(self,
                settings: dict,
                model: FederatedModel) -> None:
        """Manages the process of evaluation. Creates an instance of 
        Evaluation_Manager object, that controls all the instances
        that perform evaluation.
        -------------
        Args
            settings (dict): dictionary object cotaining all the settings of the Evaluation_Manager.
       -------------
         Returns
            None"""

        self.settings = settings
        # Sets up the flag for each available method of evaluation.
        if settings['evaluation'].get("Shapley_OR"):
            self.flag_shap_or = True
        else:
            self.flag_shap_or = False
        
        if settings['evaluation'].get("LOO_OR"):
            self.flag_loo_or = True
        else:
            self.flag_loo_or = False

        # Initialized an instance of the OR_Evaluator (One_Round Evaluator) if a flag is passed.
        if self.flag_shap_or or self.flag_loo_or:
            self.or_evaluator = OR_Evaluator(settings=settings,
                                             model=model)
    

    def track_gradients(self,
                        gradients: dict):
        """Tracks the models' gradinets for the One Round Evaluator.
        Specifically, reconstruct gradients for every possible coalition
        of interest.
        -------------
        Args
            gradients (dict): Dictionary mapping each node to its respective gradients.
       -------------
         Returns
            None"""
        if self.flag_shap_or:
            self.or_evaluator.track_shapley(gradients=gradients)
        elif self.flag_loo_or: # This is called ONLY when we don't calculate Shapley, but we calculate LOO
            self.or_evaluator.track_loo(gradients=gradients)
    

    def calculate_results(self) -> dict[str:dict] | None:
        """Calculate results for each contribution analysis method. 
        -------------
        Args
            None
       -------------
         Returns
            results | None"""
        results = {}
        if self.flag_shap_or:
            self.or_evaluator.calculate_shaply()
            results["Shapley_OneRound"] = self.or_evaluator.shapley_values
        
        if self.flag_loo_or:
            self.or_evaluator.calculate_loo()
            results["LOO_OneRound"] = self.or_evaluator.loo_values
        
        return results
