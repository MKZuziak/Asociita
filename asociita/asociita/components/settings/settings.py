from asociita.exceptions.settings.init_exception import InitMethodException, OrchestratorSettingsMissing

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
            raise InitMethodException()
    
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
        except KeyError:
            raise OrchestratorSettingsMissing
        # Nodes settings initialization
        try:
            self.nodes_settings = dict_settings['nodes']
            self.local_epochs = self.nodes_settings['local_epochs']
            self.optimizer = self.nodes_settings['model_settings']['optimizer']
            self.batch_size = self.nodes_settings['model_settings']['batch_size']
            self.lr = self.nodes_settings['model_settings']['learning_rate']
        except KeyError:
            pass
