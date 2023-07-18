from asociita.components.evaluator.or_evaluator import OR_Evaluator
from asociita.components.evaluator.lsaa_evaluator import LSAA
from asociita.components.evaluator.sample_evaluator import Sample_Evaluator
from asociita.components.evaluator.sample_evaluator import Sample_Shapley_Evaluator
from asociita.models.pytorch.federated_model import FederatedModel
from asociita.exceptions.evaluatorexception import Sample_Evaluator_Init_Exception
from asociita.utils.optimizers import Optimizers
from collections import OrderedDict
import copy
import os
import csv


class Evaluation_Manager():
    """Evaluation Manager encapsulates the whole process of assessing the marginal
    clients' contributions, so the orchestrator code is free of any encumbrance
    connected to it. Evaluation Manager life-cycle is splitted into four different
    stages: initialization, preservation, tracking and finalization. Initialization
    requires a dictionary containing all the relevant settings. Preservation should
    be called each round before the training commences (so the evaluation manager
    has last set of weights and momentum information). Tracking should be called each
    round to compute different metrics. Finalization can be called at the end of the
    life-cycle to preserve the results."""
    

    def __init__(self,
                settings: dict,
                model: FederatedModel,
                nodes: list = None,
                iterations: int = None) -> None:
        """Manages the process of evaluation. Creates an instance of Evaluation_Manager 
        object, that controls all the instances that perform evaluation. Evaluation
        Manager operatores on 'flags' that are represented as bolean attributes of the
        instance of the Class. The value of flags is dictated by the corresponding value
        used in the settings. Giving an example, settings [...]['IN_SAMPLE_LOO'] to
        True will trigger the flag of the instance. This means, that each time the method
        Evaluation_Manager().track_results is called, Evaluator will try to calculate
        Leave-One-Out score for each client in the sample.
        
        Parameters
        ----------
        settings: dict
            A dictionary containing all the relevant settings for the evaluation_manager.
        model: FederatedModel
            A initialized instance of the class asociita.models.pytorch.federated_model.FederatedModel
            This is necessary to initialize some contribution-estimation objects.
        nodes: list, default to None
            A list containing the id's of all the relevant nodes.
        iterations: int, default to None
            Number of iterations.
        
        Returns
        -------
        None

        Returns
        -------
            NotImplementedError
                If the flag for One-Round Shapley and One-Round LOO is set to True.
            Sample_Evaluator_Init_Exception
                If the Evaluation Manager is unable to initialize an instance of the 
                Sample Evaluator.
        """
        # Settings processed as new attributes
        self.settings = settings
        self.previous_c_model = None
        self.updated_c_model = None
        self.previous_optimizer = None
        self.nodes = nodes
        
        # Sets up a flag for each available method of evaluation.
        # Flag: Shapley-OneRound Method
        self.compiled_flags = []
        if settings.get("Shapley_OR"):
            self.flag_shap_or = True
            self.compiled_flags.append('shapley_or')
            raise NotImplementedError # TODO
        else:
            self.flag_shap_or = False
        # Flag: LOO-OneRound Method
        if settings.get("LOO_OR"):
            self.flag_loo_or = True
            self.compiled_flags.append('loo_or')
            raise NotImplementedError # TODO
        else:
            self.flag_loo_or = False
        # Flag: LOO-InSample Method
        if settings.get("IN_SAMPLE_LOO"):
            self.flag_sample_evaluator = True
            self.compiled_flags.append('in_sample_loo')
        else:
            self.flag_sample_evaluator = False
        # Flag: Shapley-InSample Method
        if settings.get("IN_SAMPLE_SHAP"):
            self.flag_samplesh_evaluator = True
            self.compiled_flags.append('in_sample_shap')
        else:
            self.flag_samplesh_evaluator = False
        # Flag: LSAA
        if settings.get("LSAA"):
            self.flag_lsaa_evaluator = True
            self.compiled_flags.append('LSAA')
        else:
            self.flag_lsaa_evaluator = False
        
        # Sets up a flag for each available method of score preservation
        # Flag: Preservation of partial results (for In-Sample Methods)
        if settings['preserve_evaluation'].get("preserve_partial_results"):
            self.preserve_partial_results = True
        else:
            self.preserve_partial_results = False
        # Flag: Preservation of the final result (for In-Sample Methods)
        if settings['preserve_evaluation'].get("preserve_final_results"):
            self.preserve_final_results = True

        # Initialization of objects necessary to perform evaluation.
        # Initialization: LOO-InSample Method and Shapley-InSample Method
        if self.flag_shap_or or self.flag_loo_or:
            self.or_evaluator = OR_Evaluator(settings=settings,
                                             model=model)
        # Initialization: LOO-InSample Method
        if self.flag_sample_evaluator == True:
            try:
                self.sample_evaluator = Sample_Evaluator(nodes=nodes, iterations=iterations)
            except NameError as e:
                raise Sample_Evaluator_Init_Exception
        # Initialization: Shapley-InSample Method
        if self.flag_samplesh_evaluator == True:
            try:
                self.samplesh_evaluator = Sample_Shapley_Evaluator(nodes = nodes, iterations=iterations)
            except NameError as e:
                raise Sample_Evaluator_Init_Exception
        if self.flag_lsaa_evaluator == True:
            try:
                self.lsaa_evaluator = LSAA(nodes = nodes, iterations = iterations)
                self.search_length = settings['line_search_length']
            except NameError as e:
                raise #TODO: Custom error
            except KeyError as k:
                raise #TODO: Lacking configuration error


        # Sets up the scheduler
        if settings.get("scheduler"):
            self.scheduler = settings['scheduler']
        else:
            self.scheduler = {flag: [iteration for iteration in range(iterations)] for flag in self.compiled_flags}
        
        # Auxiliary
        # Option to enter full debug mode
        if settings.get("full_debug"):
            self.full_debug = True
            if settings.get("debug_file_path"):
                self.full_debug_path = settings["full_debug_path"]
            else:
                self.full_debug_path = os.getcwd()
        else:
            self.full_debug = False
        
        # Option to disable multiprocessing features
        if settings.get("disable_multiprocessing"):
            self.multip = False
        else:
            self.multip = True
            self.number_of_workers = settings['number_of_workers']
    

    def preserve_previous_model(self,
                                previous_model: FederatedModel):
        """Preserves the model from the previous round by copying 
        its structure and using it as an attribute's value. Should
        be called each training round before the proper training
        commences.
        
        Parameters
        ----------
        previous_model: FederatedModel
            An instance of the FederatedModel object.
        Returns
        -------
        None
        """
        self.previous_c_model = copy.deepcopy(previous_model)
    

    def preserve_updated_model(self,
                               updated_model: FederatedModel):
        """Preserves the updated version of the central model
        by copying its structure and using it as an attribute's value. 
        Should be called each training after updating the weights
        of the central model.
        
        Parameters
        ----------
        updated_model: FederatedModel
            An instance of the FederatedModel object.
        Returns
        -------
        None
       """
        self.updated_c_model = copy.deepcopy(updated_model)
    
    def preserve_previous_optimizer(self,
                                    previous_optimizer: Optimizers):
        """Preserves the Optimizer from the previous round by copying 
        its structure and using it as an attribute's value. Should
        be called each training round before the proper training
        commences.
        
        Parameters
        ----------
        previous_optimizer: Optimizers
            An instance of the asociita.Optimizers class.
        Returns
        -------
        None
        """
        self.previous_optimizer = copy.deepcopy(previous_optimizer)

    
    def track_results(self,
                        gradients: OrderedDict,
                        nodes_in_sample: list,
                        iteration: int):
        """Method used to track_results after each training round.
        Because the Orchestrator abstraction should be free of any
        unnecessary encumbrance, the Evaluation_Manager.track_results()
        will take care of any result preservation and score calculation that 
        must be done in order to establish the results.
        
        Parameters
        ----------
        gradients: OrderedDict
            An OrderedDict containing gradients of the sampled nodes.
        nodes_in_sample: list
            A list containing id's of the nodes that were sampled.
        iteration: int
            The current iteration.
        Returns
        -------
        None
        """
        # Shapley-OneRound Method tracking
        # LOO-OneRound Method tracking
        if self.flag_shap_or:
            self.or_evaluator.track_shapley(gradients=gradients)
        elif self.flag_loo_or: # This is called ONLY when we don't calculate Shapley, but we calculate LOO
            self.or_evaluator.track_loo(gradients=gradients)
        
        # LOO-InSample Method
        if self.flag_sample_evaluator:
            if iteration in self.scheduler['in_sample_loo']: # Checks scheduler
                debug_values = self.sample_evaluator.update_psi(gradients = gradients,
                                            nodes_in_sample = nodes_in_sample,
                                            iteration = iteration,
                                            optimizer = self.previous_optimizer,
                                            final_model = self.updated_c_model,
                                            previous_model= self.previous_c_model)
                # Preserving debug values (if enabled)
                if self.full_debug:
                    if iteration == 0:
                        with open(os.path.join(self.full_debug_path, 'col_values_debug_loo.csv'), 'a+', newline='') as csv_file:
                            field_names = ['coalition', 'value', 'iteration']
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerow(field_names)
                            for col, value in debug_values.items():
                                csv_writer.writerow([col, value, iteration])
                    else:
                        with open(os.path.join(self.full_debug_path, 'col_values_debug_loo.csv'), 'a+', newline='') as csv_file:
                            csv_writer = csv.writer(csv_file)
                            for col, value in debug_values.items():
                                csv_writer.writerow([col, value, iteration])

        # Shapley-InSample Method
        if self.flag_samplesh_evaluator:
            if iteration in self.scheduler['in_sample_shap']: # Checks scheduler
                if self.multip:
                    debug_values = self.samplesh_evaluator.update_shap_multip(gradients = gradients,
                                                               nodes_in_sample = nodes_in_sample,
                                                               iteration = iteration,
                                                               optimizer = self.previous_optimizer,
                                                               previous_model = self.previous_c_model,
                                                               return_coalitions = self.full_debug,
                                                               number_of_workers = self.number_of_workers)
                else:
                    debug_values = self.samplesh_evaluator.update_shap(gradients = gradients,
                                                        nodes_in_sample = nodes_in_sample,
                                                        iteration = iteration,
                                                        optimizer = self.previous_optimizer,
                                                        previous_model = self.previous_c_model,
                                                        return_coalitions = self.full_debug)

                # Preserving debug values (if enabled)
                if self.full_debug:
                    if iteration == 0:
                        with open(os.path.join(self.full_debug_path, 'col_values_debug.csv'), 'a+', newline='') as csv_file:
                            field_names = ['coalition', 'value', 'iteration']
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerow(field_names)
                            for col, value in debug_values.items():
                                csv_writer.writerow([col, value, iteration])
                    else:
                        with open(os.path.join(self.full_debug_path, 'col_values_debug.csv'), 'a+', newline='') as csv_file:
                            csv_writer = csv.writer(csv_file)
                            for col, value in debug_values.items():
                                csv_writer.writerow([col, value, iteration])
    
        #LSAA Method
        if self.flag_lsaa_evaluator:
            if iteration in self.scheduler['LSAA']: # Checks scheduler
                debug_values = self.lsaa_evaluator.update_lsaa(gradients = gradients,
                                    nodes_in_sample = nodes_in_sample,
                                    iteration = iteration,
                                    search_length = self.search_length,
                                    optimizer = self.previous_optimizer,
                                    final_model = self.updated_c_model,
                                    previous_model = self.previous_c_model)
            
                            # Preserving debug values (if enabled)
                if self.full_debug:
                    if iteration == 0:
                        with open(os.path.join(self.full_debug_path, 'col_values_debug_lsaa.csv'), 'a+', newline='') as csv_file:
                            field_names = ['coalition', 'value', 'iteration']
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerow(field_names)
                            for col, value in debug_values.items():
                                csv_writer.writerow([col, value, iteration])
                    else:
                        with open(os.path.join(self.full_debug_path, 'col_values_debug_lsaa.csv'), 'a+', newline='') as csv_file:
                            csv_writer = csv.writer(csv_file)
                            for col, value in debug_values.items():
                                csv_writer.writerow([col, value, iteration])


    def finalize_tracking(self,
                          path: str = None):
        """Method used to finalize the tracking at the end of the training.
        Because the Orchestrator abstraction should be free of any
        unnecessary encumbrance, all the options configuring the behaviour
        of the finalize_tracking method, should be pre-configured at the stage
        of the Evaluation_Manager instance initialization.  
        
        Parameters
        ----------
        path: str, default to None
            a string or Path-like object to the directory in which results
            should be saved.
        Returns
        -------
        None
        """
        results = {'partial': {}, 'full': {}}

        if self.flag_shap_or:
            raise NotImplementedError
        
        if self.flag_loo_or:
            raise NotImplementedError
        
        if self.flag_sample_evaluator:
            partial_psi, psi = self.sample_evaluator.calculate_final_psi()
            results['partial']['partial_psi'] = partial_psi
            results['full']['psi'] = psi
        
        if self.flag_samplesh_evaluator:
            partial_shap, shap = self.samplesh_evaluator.calculate_final_shap()
            results['partial']['partial_shap'] = partial_shap
            results['full']['shap'] = shap
        
        if self.lsaa_evaluator:
            partial_lsaa, lsaa = self.lsaa_evaluator.calculate_final_lsaa()
            results['partial']['partial_lsaa'] = partial_lsaa
            results['full']['lsaa'] = lsaa
        
        if self.preserve_partial_results == True:
            for metric, values in results['partial'].items():
                s_path = os.path.join(path, (str(metric) + '.csv'))
                field_names = self.nodes
                field_names.append('iteration') # Field names == nodes id's (keys)
                with open(s_path, 'w+', newline='') as csv_file:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
                    csv_writer.writeheader()
                    for iteration, row in values.items():
                        row['iteration'] = iteration
                        csv_writer.writerow(row)
        
        if self.preserve_final_results == True:
            for metric, values in results['full'].items():
                s_path = os.path.join(path, (str(metric) + '.csv'))
                field_names = values.keys() # Field names == nodes id's (keys)
                with open(s_path, 'w+', newline='') as csv_file:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
                    csv_writer.writeheader()
                    csv_writer.writerow(values)
        return results