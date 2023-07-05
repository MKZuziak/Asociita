from asociita.exceptions.settingexception import SettingsObjectException

class Settings():
    def __init__(self,
                 allow_default: bool,
                 initialization_method: str = 'dict',
                 dict_settings: dict = None,
                 **kwargs) -> None:
        """Initialization of an instance of the Settings object. Requires choosing the initialization method.
        Can be initialized either from a dictionary containing all the relevant key-words or from the 
        **kwargs. It is highly advised that the Settings object should be initialized from the dicitonary.
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
        self.allow_defualt = allow_default
        if initialization_method == 'dict':
            self.init_from_dict(dict_settings=dict_settings)
            if self.enable_archiver:
                self.init_archiver_from_dict(dict_settings=dict_settings)
        elif initialization_method == 'kwargs':
            self.init_from_dict(kwargs)
        else:
            raise SettingsObjectException('Initialization method is not supported. '\
                                          'Supported methods: dict, kwargs')
    
    def init_from_dict(self,
                       dict_settings : dict):
        """Initialization of an instance of the Settings object. If the self.allow_default 
        flag was set to True during instance creation, a default object will be created in
        absence of the original values. Otherwise, the Setting object will load values passed
        in the dictionary.
        ----------
        dict_settings: dict, default to None
            A dictionary containing all the relevant settings if the initialization is made from dir. 
            Default to None
        Returns
        -------
        None"""
        # ORCHESTRATOR SETTINGS INITIZALITION
        try:
            self.orchestrator_settings = dict_settings['orchestrator']
        except KeyError:
            if self.allow_defualt:
                self.orchestrator_settings = self.generate_default_orchestrator()
            else:
                raise SettingsObjectException("The provided orchestrator settings are incomplete. The orchestrator settings should contain " \
                "the following parameters: the number of iterations ('iterations': int), number of nodes ('number_of_nodes': int), " \
                "local warm start ('local_warm_start': bool), sample size ('sample_size':int) "\
                "and whether to enable the archiver ('enable_archiver': bool), optimizer ('enable_optimizer': bool) and evaluator ('enable_evaluator': bool).")
        
        # Particular Orchestrator value retrieval
        # Number of (global)
        try:
            self.iterations = self.orchestrator_settings['iterations']
        except KeyError:
            if self.allow_defualt:
                self.iterations = 10
                print("WARNING! Number of Iterations was set to 10 as default value")
            else:
                raise SettingsObjectException("Orchestrator setting is lacking the number of iterations!")
        
        # Number of nodes
        try:
            self.number_of_nodes = self.orchestrator_settings['number_of_nodes']
        except KeyError:
            if self.allow_defualt:
                self.number_of_nodes = 10
                print("WARNING! Number of nodes was set to 10 as default value")
            else:
                raise SettingsObjectException("Orchestrator setting is lacking the number of nodes!")
        
        # Local warm start
        try:
            self.local_warm_start = self.orchestrator_settings['local_warm_start']
        except KeyError:
            if self.allow_defualt:
                self.local_warm_start = False
                print("WARNING! The default local warm start was set to 10 as default value")
            else:
                raise SettingsObjectException("Orchestrator setting is lacking the local warm start value!")

        # Sample size
        try:
            self.sample_size = self.orchestrator_settings['sample_size']
        except KeyError:
            if self.allow_defualt:
                self.sample_size = 5
                print("WARNING! The default sample size was set to 10 as default value")
            else:
                raise SettingsObjectException("Orchestrator setting is lacking the sample size value!")
        
        # Enable archiver
        try:
            self.enable_archiver = self.orchestrator_settings['enable_archiver']
        except KeyError:
            if self.allow_defualt:
                self.enable_archiver = False
                print("WARNING! The archiver was disabled as a default setting.")
            else:
                raise SettingsObjectException("Orchestrator settings are lacking the enable_archiver value.")
        
        # NODES SETTINGS INITIALIZATION
        try:
            self.nodes_settings = dict_settings['nodes']
        except KeyError:
            if self.allow_defualt:
                self.nodes_settings = self.generate_default_node()
            else:
                raise SettingsObjectException("The provided nodes settings are incomplete. The nodes settings should contain " \
                "the following parameters: the number of local epochs ('local_epochs': int), optimizer ('optimizer': str), " \
                "batch size ('batch_size': str) and learning rate ('learning rate' : float).")
        
        # Particular nodes value retrieval
        try:
            self.local_epochs = self.nodes_settings['local_epochs']
        except KeyError:
            if self.allow_defualt:
                self.local_epochs = 2
                self.nodes_settings["locla_epochs"] = 2
            else:
                raise SettingsObjectException("The provided nodes settings are incomplete. The nodes settings should contain " \
                "the following parameters: the number of local epochs ('local_epochs': int), optimizer ('optimizer': str), " \
                "batch size ('batch_size': str) and learning rate ('learning rate' : float).")
        
        
        # MODEL SETTINGS INITIALIZATION
        try:
            self.nodes_settings['model_settings']
        except KeyError:
            if self.allow_defualt:
                self.model_settings = self.generate_default_model()
                self.nodes_settings['model_settings'] = self.model_settings # Appedig model settings to nodes_settings
            else:
                raise SettingsObjectException("The provided nodes settings are incomplete. The nodes settings should contain " \
                "the following parameters: the number of local epochs ('local_epochs': int), optimizer ('optimizer': str), " \
                "batch size ('batch_size': str) and learning rate ('learning rate' : float).")
        
        try:
            self.optimizer = self.nodes_settings['model_settings']['optimizer']
        except KeyError:
            if self.allow_defualt:
                self.optimizer = "RMS"
                self.nodes_settings['model_settings']['optimizer'] = 'RMS'
            else:
                raise SettingsObjectException("The provided nodes settings are incomplete. The nodes settings should contain " \
                "the following parameters: the number of local epochs ('local_epochs': int), optimizer ('optimizer': str), " \
                "batch size ('batch_size': str) and learning rate ('learning rate' : float).")
        
        try:
            self.batch_size = self.nodes_settings['model_settings']['batch_size']
        except KeyError:
            if self.allow_defualt:
                self.batch_size = 32
                self.nodes_settings['model_settings']['batch_size'] = 32
            else:
                raise SettingsObjectException("The provided nodes settings are incomplete. The nodes settings should contain " \
                "the following parameters: the number of local epochs ('local_epochs': int), optimizer ('optimizer': str), " \
                "batch size ('batch_size': str) and learning rate ('learning rate' : float).")
        
        try:
            self.lr = self.nodes_settings['model_settings']['learning_rate']
        except KeyError:
            if self.allow_defualt:
                self.lr = 0.001
                self.nodes_settings['model_settings']['learning_rate'] = 0.001
            else:
                raise SettingsObjectException("The provided nodes settings are incomplete. The nodes settings should contain " \
                "the following parameters: the number of local epochs ('locla_e`pochs': int), optimizer ('optimizer': str), " \
                "batch size ('batch_size': str) and learning rate ('learning rate' : float).")
        
        self.print_orchestrator_template()
        self.print_nodes_template()
        
    def generate_default_orchestrator(self) -> dict:
        """Generates default orchestrator template.
        ----------
        None
        Returns
        -------
        dict"""
        print("WARNING! Generating a new default orchestrator template.") #TODO: Switch for logger
        orchestrator = dict()
        orchestrator["iterations"] = 10
        orchestrator["number_of_nodes"] = 10
        orchestrator["local_warm_start"] = False
        orchestrator["sample_size"] = 5
        orchestrator["enable_archiver"] = False
        return orchestrator


    def generate_default_node(self) -> dict:
        """Generates default node template.
        ----------
        None
        Returns
        -------
        dict"""
        print("WARNING! Generating a new default node template.") #TODO: Switch for logger
        node = dict()
        node['model_settings'] = dict()
        node['local_epochs'] = 2
        return node
    

    def generate_default_model(self) -> dict:
        """Generates default model template.
        ----------
        None
        Returns
        -------
        dict"""
        print("WARNING! Generatic a new default node model.") #TODO: Switch for logger
        model = dict()
        model['optimizer'] = 'RMS'
        model['batch_size'] = 32
        model['learning_rate'] = 0.001
        return model
    

    def print_orchestrator_template(self,
                                    orchestrator_type: str = 'general'):
        """Prints out the used template for the orchestrator.
        ----------
        orchestrator_type: str, default to general
            A type of the orchestrator
        Returns
        -------
        dict"""
        string = f"""orchestrator_type: {orchestrator_type},
        iterations: {self.iterations},
        number_of_nodes: {self.number_of_nodes},
        local_warm_start: {self.local_warm_start},
        sample_size:    {self.sample_size},
        enable_archiver: {self.enable_archiver}                    
        """
        print(string) #TODO: Switch for logger
    

    def print_nodes_template(self):
        """Prints out the used template for the nodes.
        ----------
        None
        Returns
        -------
        dict"""
        string = f"""local_epochs: {self.local_epochs},
        optimizer: {self.optimizer},
        batch_size: {self.batch_size},
        learning_rate: 0.001                    
        """
        print(string) #TODO: Switch for logger