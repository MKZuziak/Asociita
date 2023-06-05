from asociita.components.evaluator.or_evaluator import OR_Evaluator
from asociita.components.evaluator.sample_evaluator import Sample_Evaluator
from asociita.models.pytorch.federated_model import FederatedModel
from asociita.exceptions.evaluatorexception import Sample_Evaluator_Init_Exception
import copy
import os
import csv


class Evaluation_Manager():
    def __init__(self,
                settings: dict,
                model: FederatedModel,
                nodes: list = None,
                iterations: int = None) -> None:
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
        self.previous_c_model = None
        self.updated_c_model = None
        self.previous_optimizer = None
        # Sets up the flag for each available method of evaluation.
        if settings['evaluation'].get("Shapley_OR"):
            self.flag_shap_or = True
            raise NotImplementedError # TODO
        else:
            self.flag_shap_or = False
        
        if settings['evaluation'].get("LOO_OR"):
            self.flag_loo_or = True
            raise NotImplementedError # TODO
        else:
            self.flag_loo_or = False
        
        if settings['evaluation'].get("IN_SAMPLE_LOO"):
            self.flag_sample_evaluator = True
        else:
            self.flag_sample_evaluator = False


        # Initialized an instance of the OR_Evaluator (One_Round Evaluator) if a flag is passed.
        if self.flag_shap_or or self.flag_loo_or:
            self.or_evaluator = OR_Evaluator(settings=settings,
                                             model=model)
        elif self.flag_sample_evaluator == True:
            try:
                self.sample_evaluator = Sample_Evaluator(nodes=nodes, iterations=iterations)
            except NameError as e:
                raise Sample_Evaluator_Init_Exception
    

    def preserve_previous_model(self,
                                previous_model):
        self.previous_c_model = copy.deepcopy(previous_model)
    
    def preserve_updated_model(self,
                               updated_model):
        self.updated_c_model = copy.deepcopy(updated_model)
    
    def preserve_previous_optimizer(self,
                                    previous_optimizer):
        self.previous_optimizer = copy.deepcopy(previous_optimizer)

    
    def track_results(self,
                        gradients: dict,
                        nodes_in_sample: list,
                        iteration: int,
                        optimizer):
        """Tracks the models' gradinets.
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
        
        if self.flag_sample_evaluator:
            self.sample_evaluator.update_psi(gradients = gradients,
                                        nodes_in_sample = nodes_in_sample,
                                        iteration = iteration,
                                        optimizer = optimizer,
                                        final_model = self.updated_c_model,
                                        previous_model= self.previous_c_model)
    

    def finalize_tracking(self):
        if self.flag_shap_or:
            raise NotImplementedError
        
        if self.flag_loo_or:
            raise NotImplementedError
        
        if self.flag_sample_evaluator:
            psi = self.sample_evaluator.calculate_final_psi()
            return psi

    # def calculate_results(self,
    #                       map_results:bool = True) -> dict[str:dict] | None:
    #     """Calculate results for each contribution analysis method. 
        
    #     -------------
    #     Args
    #         map_results (bool): If set to true, the method will map results
    #             of the evaluation to each node and will return an additionall
    #             dictionary of the format: {node_id: {method:score, method:score}}
    #    -------------
    #      Returns
    #         results | None"""
    #     results = {}
    #     if map_results:
    #         mapped_results = {node: {'node_number': node} for node in self.settings['nodes']}
        
    #     if self.flag_shap_or:
    #         self.or_evaluator.calculate_shaply()
    #         results["Shapley_OneRound"] = self.or_evaluator.shapley_values
    #         if map_results:
    #             for node, result in zip(mapped_results, results['Shapley_OneRound'].values()):
    #                 mapped_results[node]["shapley_one_round"] = result

        
    #     if self.flag_loo_or:
    #         self.or_evaluator.calculate_loo()
    #         results["LOO_OneRound"] = self.or_evaluator.loo_values
    #         if map_results:
    #             for node, result in zip(mapped_results, results['LOO_OneRound'].values()):
    #                 mapped_results[node]['leave_one_out_one_round'] = result
        
    #     if mapped_results:
    #         return (results, mapped_results)
    #     else:
    #         return results


    # def save_results(self,
    #                  path: str,
    #                  mapped_results: str,
    #                  file_name: str = 'contribution_results') -> None:
    #     """Preserves metrics of every contribution index calculated by the manager
    #     in a csv file.
    #     -------------
    #     Args
    #         path (str): a path to the directory in which the file should be saved.
    #         mapped_results: results mapped to each note, stored in a dictionary
    #             of a format {node_id: {method:score, method:score}
    #         file_name (str): a proper filename with .csv ending. Default is
    #             "contribution.results.csv
    #    -------------
    #      Returns
    #         results | None"""
    #     file_name = 'contribution_results.csv'
    #     path = os.path.join(path, file_name)
    #     with open(path, 'w', newline='') as csvfile:
    #         header = [column_name for column_name in mapped_results[0].keys()]
    #         writer = csv.DictWriter(csvfile, fieldnames=header)

    #         writer.writeheader()
    #         for row in mapped_results.values():
    #             writer.writerow(row)