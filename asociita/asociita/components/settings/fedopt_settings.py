from asociita.exceptions.settingexception import SettingsObjectException
from asociita.components.settings.settings import Settings
import os
import time

class FedoptSettings(Settings):
    def __init__(self, 
                 allow_default: bool, 
                 initialization_method: str = 'dict', 
                 dict_settings: dict = None, 
                 **kwargs) -> None:
        """Initialization of an instance of the FedoptSettings object. Requires choosing the initialization method.
        Can be initialized either from a dictionary containing all the relevant key-words or from the 
        **kwargs. It is highly advised that the Settings object should be initialized from the dicitonary.
        It inherits all the properties and attributes from the Parent class addting an additional Optimizer settings.
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
            self.init_optimizer_from_dict(dict_settings=self.orchestrator_settings)
        else: # Initialization from **kwargs
            self.init_optimizer_from_kwargs(kwargs)


    def init_optimizer_from_dict(self,
                                 dict_settings: dict):
        """Loads the optimizer configuration onto the settings instance. If the self.allow_default 
        flag was set to True during instance creation, a default archiver tempalte will be created.
        ----------
        dict_settings: dict, default to None
            A dictionary containing all the relevant settings if the initialization is made from dir. 
            Default to None
        Returns
        -------
        None"""
        try:
            self.optimizer_settings = dict_settings['optimizer']
        except KeyError:
            if self.allow_defualt:
                self.optimizer_settings = self.generate_default_optimizer()
            else:
                raise SettingsObjectException("Optimizer was enabled, but the optimizer settings are missing and the" \
                                              "allow_default flag was set to False. Please provide archiver settings or"\
                                                "set the allow_default flag to True or disable the archiver.")


        assert self.optimizer_settings['name'], SettingsObjectException("Optimizer name is missing!")
        # Sanity check for the optimizer
        try:
            self.optimizer_settings['learning_rate'] = self.optimizer_settings['learning_rate']
        except KeyError:
            if self.allow_defualt:
                self.optimizer_settings['learning_rate'] = 1.00
                print("WARNING! Central optimizer lr was set to 1.00 by default.")
            else:
                raise SettingsObjectException("Optimizer object is missing the key properties!")
        
        if self.optimizer_settings['name'] == "FedAdagard":
            assert self.optimizer_settings['b1'], SettingsObjectException("FedAdagard requires b1 value!")
            assert self.optimizer_settings['tau'], SettingsObjectException("FedAdagard requires tau value!")
        
        if self.optimizer_settings['name'] == "FedAdam" or self.optimizer_settings['name'] == 'FedYogi':
            assert self.optimizer_settings['b1'], SettingsObjectException("FedAdam or Fedyogi requires b1 value!")
            assert self.optimizer_settings['b2'], SettingsObjectException("FedAdam or FedYogi requires b2 balue!")
            assert self.optimizer_settings['tau'], SettingsObjectException("FedAdam or FedYogi requires tau value!")
        
        self.orchestrator_settings['optimizer'] = self.optimizer_settings # Attachings archiver settings to orchestrator settings.
        self.print_optimizer_template()


    def generate_default_optimizer(self):
        """Generates default optimizer template.
        ----------
        None
        Returns
        -------
        dict"""
        print("WARNING! Generatic a new default optimizer template.") #TODO: Switch for logger
        optimizer = dict()
        optimizer['name'] = 'Simple'
        optimizer['learning_rate'] = 1.00 #Equal to FedAvg
        return optimizer


    def print_optimizer_template(self):
        """Prints out the used template for the optimizer.
        ----------
        None
        Returns
        -------
        dict"""
        if self.optimizer_settings['name'] == 'Simple':
            string = f"""name: {self.optimizer_settings['name']},
            learning_rate : {self.optimizer_settings['learning_rate']},
            """
        elif self.optimizer_settings['name'] == "FedAdagard":
            string = f"""name : {self.optimizer_settings['name']},
            learning_rate : {self.optimizer_settings['learning_rate']},
            b1 : {self.optimizer_settings['b1']},
            tau : {self.optimizer_settings['tau']}
            """
        else: # FedYogi or FedAdam
            string = f"""name : {self.optimizer_settings['name']},
            learning_rate : {self.optimizer_settings['learning_rate']},
            b1 : {self.optimizer_settings['b1']},
            b2: {self.optimizer_settings['b2']}
            tau : {self.optimizer_settings['tau']}
            """

        print(string) #TODO: Switch for logger