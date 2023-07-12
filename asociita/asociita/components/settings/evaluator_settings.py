from asociita.components.settings.fedopt_settings import FedoptSettings
from asociita.exceptions.settingexception import SettingsObjectException


class Evaluator_Settings(FedoptSettings):
    def __init__(self, allow_default: bool, 
                 initialization_method: str = 'dict', 
                 dict_settings: dict = None, 
                 **kwargs) -> None:
        """Initialization of an instance of the Evaluator object. Requires choosing the initialization method.
        Can be initialized either from a dictionary containing all the relevant key-words or from the 
        **kwargs. It is highly advised that the Settings object should be initialized from the dicitonary.
        It inherits all the properties and attributes from the Parent class adding additionally the Evaluator object.
        Parameters
        ----------
        allow_default: bool
            A logical switch to allow using default values in absence of passed values.
        initialization_method: str, default to 'dict' 
            The method of initialization. Either 'dict' or 'kwargs'.
        dict_settings: dict, default to None
            A dictionary containing all the relevant settings if the initialization is made from dir. 
            Default to None
        Returns
        -------
        None"""
        super().__init__(allow_default, 
                         initialization_method, 
                         dict_settings, 
                         **kwargs)
        if initialization_method == 'dict':
            self.init_evaluator_from_dict(dict_settings=self.orchestrator_settings)
        else: # Initialization from **kwargs
            self.init_evaluator_from_kwargs(kwargs)
    

    def init_evaluator_from_dict(self,
                                dict_settings: dict):
        """Loads the evaluator configuration onto the settings instance. If the self.allow_default 
        flag was set to True during instance creation, a default evaluator tempalte will be created
        in absence of the one provided.
        ----------
        dict_settings: dict, default to None
            A dictionary containing all the relevant settings if the initialization is made from dir. 
            Default to None
        Returns
        -------
        None"""
        try:
            self.evaluator_settings = dict_settings['evaluator']
        except KeyError:
            if self.allow_defualt:
                self.evaluator_settings = self.generate_default_evaluator()
            else:
                raise SettingsObjectException("Evaluator was enabled, but the evaluator settings are missing and the" \
                                              "allow_default flag was set to False. Please provide evaluator settings or"\
                                                "set the allow_default flag to True or disable the evaluator.")
        
        # Sanity check for the evaluator
        try:
            self.evaluator_settings["LOO_OR"]
        except KeyError:
            if self.allow_defualt:
                self.evaluator_settings["LOO_OR"] = False
                print("WARNING! Leave-one-out One-Round was disabled by default.")
            else:
                raise SettingsObjectException("Evaluator object is missing the key properties!")
        
        try:
            self.evaluator_settings["Shapley_OR"]
        except KeyError:
            if self.allow_defualt:
                self.evaluator_settings["Shapley_OR"] = False
                print("WARNING! Shapley One-Round was disabled by default.")
            else:
                raise SettingsObjectException("Evaluator object is missing the key properties!")
        
        try:
            self.evaluator_settings["IN_SAMPLE_LOO"]
        except KeyError:
            if self.allow_defualt:
                self.evaluator_settings["IN_SAMPLE_LOO"] = False
                print("WARNING! In-sample Shapley was disabled by default.")
            else:
                raise SettingsObjectException("Evaluator object is missing the key properties!")
        
        try:
            self.evaluator_settings["IN_SAMPLE_SHAP"]
        except KeyError:
            if self.allow_defualt:
                self.evaluator_settings["IN_SAMPLE_SHAP"] = False
                print("WARNING! In-sample Shapley was disabled by default.")
            else:
                raise SettingsObjectException("Evaluator object is missing the key properties!")
        
        try:
            self.evaluator_settings["preserve_evaluation"]
        except KeyError:
            if self.allow_defualt:
                self.evaluator_settings["preserve_evaluation"] = False
                print("WARNING! Preserve-evaluation option was disabled by default.")
            else:
                raise SettingsObjectException("Evaluator object is missing the key properties!")
        
        try:
            self.evaluator_settings["full_debug"]
        except KeyError:
            if self.allow_defualt:
                self.evaluator_settings["full_debug"] = False
                print("WARNING! Preserve-evaluation option was disabled by default.")
            else:
                raise SettingsObjectException("Evaluator object is missing the key properties!")
        
        self.orchestrator_settings['evaluator'] = self.evaluator_settings # Attaching evaluator to the orchestrator_settings

        self.print_evaluator_template()
        
    
    def generate_default_evaluator(self):
        """Generates default optimizer template.
        ----------
        None
        Returns
        -------
        dict"""
        print("WARNING! Generatic a new default archiver template.") #TODO: Switch for logger
        evaluator = dict()
        evaluator['LOO_OR'] = False
        evaluator["Shapley_OR"] = False
        evaluator["IN_SAMPLE_LOO"] = True
        evaluator["IN_SAMPLE_SHAP"] = False
        evaluator["preserve_evaluation"] = {
            "preserve_partial_results": True,
            "preserve_final_results": True
        }
        evaluator["full_debug"] = False
        evaluator["number_of_workers"] = 50

        return evaluator


    def print_evaluator_template(self):
        """Prints out the used template for the evaluator.
        ----------
        None
        Returns
        -------
        dict"""
        string = f"""
        Enable One-Round Leave-one-out: {self.evaluator_settings['LOO_OR']},
        Enable One-Round Shapley: {self.evaluator_settings['Shapley_OR']},
        Enable In-Sample Leave-one-out: {self.evaluator_settings['IN_SAMPLE_LOO']},
        Enable In-Sample Shapley: {self.evaluator_settings['IN_SAMPLE_SHAP']},
        Preserve evaluation: {self.evaluator_settings["preserve_evaluation"]},
        Enable full debug mode: {self.evaluator_settings["full_debug"]}
        """
        print(string) #TODO: Switch for logger