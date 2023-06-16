from asociita.exceptions.settingexception import SettingsObjectException

class Settings():
    def __init__(self,
                 initialization_method: str = 'dict',
                 dict_settings: dict = None,
                 **kwargs) -> None:
        if initialization_method == 'dict':
            self.init_from_dict(dict_settings=dict_settings)
        elif initialization_method == 'kwargs':
            self.init_from_dict(kwargs)
        else:
            raise SettingsObjectException('Initialization method is not supported. '\
                                          'Supported methods: dict, kwargs')
    
    def init_from_dict(self,
                       dict_settings : dict):
        # Orchestrator settings initialization
        try:
            self.orchestrator_settings = dict_settings['orchestrator']
            self.iterations = self.orchestrator_settings['iterations']
            self.number_of_nodes = self.orchestrator_settings['number_of_nodes']
            self.local_warm_start = self.orchestrator_settings['local_warm_start']
            self.sample_size = self.orchestrator_settings['sample_size']
            self.metrics_save_path = self.orchestrator_settings['metrics_save_path']
            self.enable_archiver = self.orchestrator_settings['enable_archiver']
        except KeyError:
            raise SettingsObjectException("The provided orchestrator settings are incomplete. The orchestrator settings should contain " \
            "the following parameters: the number of iterations ('iterations': int), number of nodes ('number_of_nodes': int), " \
            "local warm start ('local_warm_start': bool), sample size ('sample_size':int), metrics save path ('metrics_save_path' : str or Path) "\
                "and whether to enable the archiver ('enable_archiver': bool).")
        # Nodes settings initialization
        try:
            self.nodes_settings = dict_settings['nodes']
            self.model_settings = dict_settings['nodes']['model_settings']
            self.local_epochs = self.nodes_settings['local_epochs']
            self.optimizer = self.nodes_settings['model_settings']['optimizer']
            self.batch_size = self.nodes_settings['model_settings']['batch_size']
            self.lr = self.nodes_settings['model_settings']['learning_rate']
        except KeyError:
            raise SettingsObjectException("The provided orchestrator settings are incomplete. The nodes settings should contain " \
            "the following parameters: the number of local epochs ('locla_e`pochs': int), optimizer ('optimizer': str), " \
            "batch size ('batch_size': str) and learning rate ('learning rate' : float).")
        if self.enable_archiver:
            try:
                self.archiver_settings = self.orchestrator_settings['archiver_settings']
            except KeyError:
                raise SettingsObjectException('The archiver is enabled in the settings, but the init method was unable to '\
                                              'retrieve the archiver settings. Provide relevant settings packed as dictionary ' \
                                              "or disable the archiver using option <'enable_archiver': False>.")
