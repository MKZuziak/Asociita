from asociita.utils.computations import Subsets
from asociita.models.pytorch.federated_model import FederatedModel
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
